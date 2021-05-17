from dpr.models import init_biencoder_components
from dpr.utils.model_utils import setup_for_distributed_mode, get_model_obj, load_states_from_checkpoint
from argparse import Namespace
import torch
from dpr.options import add_encoder_params, setup_args_gpu, print_args, set_encoder_params_from_state, \
    add_tokenizer_params, add_cuda_params
import pickle
import numpy as np
import pandas as pd
import json
from argparse import Namespace
import argparse

def main(args2) :
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = Namespace(batch_size=32, distributed_world_size=1, device=device, do_lower_case=False, encoder_model_type='hf_bert', 
                     fp16=False, fp16_opt_level='O1', local_rank=-1, 
                     model_file=args2.model_file, 
                     n_gpu=1, no_cuda=False, num_shards=1, pretrained_file=None, 
                     pretrained_model_cfg='bert-base-uncased', projection_dim=0, sequence_length=512, shard_id=0)
    saved_state = load_states_from_checkpoint(args.model_file)
    set_encoder_params_from_state(saved_state.encoder_params, args)
    tensorizer, encoder, _ = init_biencoder_components(args.encoder_model_type, args, inference_only=True)
    encoder = encoder.question_model
    encoder, _ = setup_for_distributed_mode(encoder, None, args.device, args.n_gpu,
                                            args.local_rank,
                                            args.fp16)
    encoder.eval()
    # load weights from the model file
    model_to_load = get_model_obj(encoder)
    print('Loading saved model state ...')
    prefix_len = len('question_model.')
    question_encoder_state = {key[prefix_len:]: value for (key, value) in saved_state.model_dict.items() if
                              key.startswith('question_model.')}
    model_to_load.load_state_dict(question_encoder_state)
    
    def get_embedding(q) :
        with torch.no_grad():
            batch_token_tensors = [tensorizer.text_to_tensor(q)]
            q_ids_batch = torch.stack(batch_token_tensors, dim=0).cuda()
            q_seg_batch = torch.zeros_like(q_ids_batch).cuda()
            q_attn_mask = tensorizer.get_attn_mask(q_ids_batch)
            _, out, _ = encoder(q_ids_batch, q_seg_batch, q_attn_mask)
            return out.cpu().split(1, dim=0)[0][0].numpy()
    
    
    def filter_json_data() :
        with open(args2.embeddings_file, 'rb') as f:
            embeddings = pickle.load(f)
        with open(args2.input_file, 'r') as f :
            data=json.load(f)
        final_data = []
        for i in range(len(data)) :
            if i%1000==0 :
                print (i)
            q_embedding = get_embedding(data[i]['question'])
            if np.dot(q_embedding, embeddings[i][1]) > args2.threshold :
                final_data.append(data[i])
        print ('Total filtered data : ', len(final_data))
        with open(args2.output_file, 'w') as f:
            json.dump(final_data, f)

    def filer_tsv_data() :
        with open(args2.embeddings_file, 'rb') as f:
            embeddings = pickle.load(f)
        df = pd.read_csv(args2.input_file, sep='\t')
        questions = df['target_text'].tolist() if 'target_text' in df else df['target_text0'].tolist()
        passages = df['input_text'].tolist()
        filtered_q, filtered_p = [], []
        for i in range(len(questions)) :
            if i%1000==0 :
                print (i)
            q_embedding = get_embedding(questions[i])
            if np.dot(q_embedding, embeddings[i][1]) > args2.threshold :
                filtered_q.append(questions[i])
                filtered_p.append(passages[i])
        df = pd.DataFrame()
        df['input_text'] = pd.Series(filtered_p)
        df['target_text'] = pd.Series(filtered_q)
        print ('Total filtered data : ', len(df))
        df.to_csv(args2.output_file, sep='\t')
        
    if args2.input_type == 'tsv' :
        filer_tsv_data()
    else :
        filter_json_data()
    return
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_file', required=True, type=str)
    parser.add_argument('--embeddings_file', required=True, type=str)
    parser.add_argument('--input_file', required=True, type=str)
    parser.add_argument('--input_type', required=True, type=str)
    parser.add_argument('--output_file', required=True, type=str)
    parser.add_argument('--threshold', required=True, type=float)
    args = parser.parse_args()
    main(args)