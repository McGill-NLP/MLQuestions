#!/bin/bash

#Evaluate no-adaptation QG model on NaturalQuestions (IID) and MLQuestions (OOD) Data
cd QG/
echo "No-adaptation QG model performance on NaturalQuestions(IID) test data"
echo "---------------------------------------------------------------------"
python eval.py --checkpoint NQ-checkpoint/ --eval_file ../data/test_nq.tsv
echo "No-adaptation QG model performance on MLQuestions(OOD) test data"
echo "---------------------------------------------------------------------"
python eval.py --checkpoint NQ-checkpoint/ --eval_file ../data/test.tsv

#Evaluate no-adaptation IR model on NaturalQuestions (IID) and MLQuestions (OOD) Data
cd ../IR

echo "No-adaptation IR model performance on NaturalQuestions(IID) test data"
echo "---------------------------------------------------------------------"
python generate_dense_embeddings.py --model_file NQ-checkpoint/bert-base-encoder.cp --ctx_file ../data/test_passages_nq.tsv --out_file NQ-checkpoint/embeddings_11k_nq
python eval_retriever.py --model_file NQ-checkpoint/bert-base-encoder.cp --embeddings NQ-checkpoint/embeddings_11k_nq_0.pkl --eval_file ../data/test_nq.tsv

echo "No-adaptation IR model performance on MLQuestions(OOD) test data"
echo "---------------------------------------------------------------------"
python generate_dense_embeddings.py --model_file NQ-checkpoint/bert-base-encoder.cp --ctx_file ../data/test_passages.tsv --out_file NQ-checkpoint/embeddings_11k
python eval_retriever.py --model_file NQ-checkpoint/bert-base-encoder.cp --embeddings NQ-checkpoint/embeddings_11k_0.pkl --eval_file ../data/test.tsv