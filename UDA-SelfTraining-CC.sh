#1. QG model generates questions from 50k passages and prepares synthetic data for self-training QG
cd QG/
python generate.py --checkpoint NQ-checkpoint/ --file ../data/passages_unaligned.tsv

#2. IR model generates embeddings for 50k passages
cd ../IR/
python generate_dense_embeddings.py --model_file NQ-checkpoint/bert-base-encoder.cp --ctx_file ../data/passages_unaligned.tsv --out_file NQ-checkpoint/embeddings_50k

#3. IR model retrieves passages from questions and prepares synthetic data for self-training IR
python generate.py --model_file NQ-checkpoint/bert-base-encoder.cp --embeddings NQ-checkpoint/embeddings_50k_0.pkl --out_file outputs/ST.tsv

#4. Filter IR model data using QG model (cross-consistency)
cd ../QG/
python compute_loss.py --file ../IR/outputs/ST.tsv --checkpoint NQ-checkpoint/ --save_to NQ-checkpoint/losses.txt
python consistency.py --input_file ../IR/outputs/ST.tsv --threshold_file NQ-checkpoint/losses.txt --output_file ../IR/outputs/ST-cc.tsv --threshold 5.95

#5. Generated data is converted to json format for training IR model
cd ../IR/
python gen_dpr_data.py --input_file outputs/ST-cc.tsv --out_file outputs/ST.json

#6. Filter QG model data using IR model (cross-consistency)
python consistency.py --model_file NQ-checkpoint/bert-base-encoder.cp --embeddings_file NQ-checkpoint/embeddings_50k_0.pkl --input_file NQ-checkpoint/QG-predictions-50K.tsv --output_file NQ-checkpoint/QG-predictions-50K-cc.tsv --threshold 71.65

#7. Train QG model on synthetic self-training data
cd ../QG/
python train.py --epochs 5 --train_file NQ-checkpoint/QG-predictions-50K-cc.tsv --checkpoint NQ-checkpoint/

#8. Train IR model on synthetic self-training data
cd ../IR/
python train_dense_encoder.py --encoder_model_type hf_bert --pretrained_model_cfg bert-base-uncased --train_file outputs/ST.json --num_train_epochs 6 --model_file NQ-checkpoint/bert-base-encoder.cp --output_dir outputs/ --batch_size 32 --dev_file ../data/dev.json

#9. Evaluate QG model on test data
cd ../QG/
python eval.py --checkpoint outputs/ --eval_file ../data/test.tsv

#10. Evaluate IR model on test data
#a. Generate embeddings of 11k test passages
cd ../IR/
python generate_dense_embeddings.py --model_file outputs/dpr_biencoder.5.1106 --ctx_file ../data/test_passages.tsv --out_file outputs/embeddings_11k
#b. evaluate top-k retrieval accuracy on test data
python eval_retriever.py --model_file outputs/dpr_biencoder.5.1106 --embeddings outputs/embeddings_11k_0.pkl --eval_file ../data/test.tsv