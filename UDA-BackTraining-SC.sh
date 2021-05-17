#1. QG model generates questions from 50k passages and prepares synthetic data for back-training IR
cd QG/
python generate.py --checkpoint NQ-checkpoint/ --file ../data/passages_unaligned.tsv

#2. Filter QG model data using QG model (self-consistency)
cd ../QG/
python compute_loss.py --file NQ-checkpoint/QG-predictions-50K.tsv --checkpoint NQ-checkpoint/ --save_to NQ-checkpoint/losses.txt
python consistency.py --input_file NQ-checkpoint/QG-predictions-50K.tsv --threshold_file NQ-checkpoint/losses.txt --output_file NQ-checkpoint/QG-predictions-50K-sc.tsv --threshold 1.19

#3. Generated data is converted to json format for training IR model
cd ../IR/
python gen_dpr_data.py --input_file ../QG/NQ-checkpoint/QG-predictions-50K-sc.tsv --out_file outputs/BT.json

#4. IR model generates embeddings for 50k passages
python generate_dense_embeddings.py --model_file NQ-checkpoint/bert-base-encoder.cp --ctx_file ../data/passages_unaligned.tsv --out_file NQ-checkpoint/embeddings_50k

#5. IR model retrieves passages from questions and prepares synthetic data for back-training QG
python generate.py --model_file NQ-checkpoint/bert-base-encoder.cp --embeddings NQ-checkpoint/embeddings_50k_0.pkl --out_file ../QG/outputs/BT.tsv

#7. Train QG model on synthetic back-training data
cd ../QG/
python train.py --epochs 5 --train_file outputs/BT.tsv --checkpoint NQ-checkpoint/

#8. Train IR model on synthetic back-training data
cd ../IR/
python train_dense_encoder.py --encoder_model_type hf_bert --pretrained_model_cfg bert-base-uncased --train_file outputs/BT.json --num_train_epochs 6 --model_file NQ-checkpoint/bert-base-encoder.cp --output_dir outputs/ --batch_size 32 --dev_file ../data/dev.json

#9. Evaluate QG model on test data
cd ../QG/
python eval.py --checkpoint outputs/ --eval_file ../data/test.tsv

#10. Evaluate IR model on test data
#a. Generate embeddings of 11k test passages
cd ../IR/
python generate_dense_embeddings.py --model_file outputs/dpr_biencoder.5.1581 --ctx_file ../data/test_passages.tsv --out_file outputs/embeddings_11k
#b. evaluate top-k retrieval accuracy on test data
python eval_retriever.py --model_file outputs/dpr_biencoder.5.1581 --embeddings outputs/embeddings_11k_0.pkl --eval_file ../data/test.tsv