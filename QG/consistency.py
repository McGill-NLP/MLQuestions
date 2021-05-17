import numpy as np
import pandas as pd
from argparse import Namespace
import argparse

def main(args) :
    df = pd.read_csv(args.input_file, sep='\t')
    qs = df['target_text'].tolist() if 'target_text' in df else df['target_text0'].tolist()
    ps = df['input_text'].tolist()
    fp, fq = [], []
    thresholds = open(args.threshold_file, 'r').readlines()
    thresholds = [float(t[:-1]) for t in thresholds]
    print (len(df), len(thresholds))
    for i in range(len(thresholds)) :
        if thresholds[i] < args.threshold :
            fp.append(ps[i])
            fq.append(qs[i])
    df = pd.DataFrame()
    df['input_text'] = pd.Series(fp)
    df['target_text'] = pd.Series(fq)
    df.to_csv(args.output_file, sep='\t')
    print (len(df))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', required=True, type=str)
    parser.add_argument('--threshold_file', required=True, type=str)
    parser.add_argument('--output_file', required=True, type=str)
    parser.add_argument('--threshold', required=True, type=float)
    args = parser.parse_args()
    main(args)