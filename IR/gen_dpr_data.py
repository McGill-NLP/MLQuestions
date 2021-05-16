import glob
import json
import pandas as pd
from rank_bm25 import BM25Okapi
import argparse

def get_negative_ctxts(question, passage, k, corpus, bm25) :
    passages = bm25.get_top_n(question.split(), corpus, n=k+1)
    if passage in passages :
        passages.remove(passage)
    return passages[:k]

def main(args) :
    qg_aug_data = pd.read_csv(args.input_file, sep='\t')

    corpus = pd.read_csv('../data/passages_unaligned.tsv', sep='\t')['input_text'].tolist()
    tokenized_corpus = [doc.split(" ") for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)

    laques_data = []
    paras, questions = qg_aug_data['input_text'].tolist(), qg_aug_data['target_text'].tolist()
    for i in range(len(paras)) :
        if i%500 == 0 :
            print ('Completed : ', i)
        datum = {
                 'question': questions[i], 
                 'positive_ctxs': [{'text': paras[i]}], 
                 'negative_ctxs': [],
                 'hard_negative_ctxs': [{'text': passage} for passage in get_negative_ctxts(questions[i], paras[i], 7, corpus, bm25)]
                }
        laques_data.append(datum)

    json_string = json.dumps(laques_data)
    with open(args.out_file, 'w') as f:
        json.dump(laques_data, f)
    print (len(laques_data))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', required=True, type=str)
    parser.add_argument('--out_file', required=True, type=str)
    args = parser.parse_args()
    main(args)