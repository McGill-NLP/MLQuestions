from dpr.models import init_biencoder_components
from dpr.utils.model_utils import setup_for_distributed_mode, get_model_obj, load_states_from_checkpoint
from argparse import Namespace
import torch
from dpr.options import add_encoder_params, setup_args_gpu, print_args, set_encoder_params_from_state, \
    add_tokenizer_params, add_cuda_params
import pickle
import numpy as np
import pandas as pd
import argparse
import scipy.stats as ss

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
        
    with open(args2.embeddings, 'rb') as f:
        embeddings = pickle.load(f)
    embeddings = np.array([e[1] for e in embeddings])
    df = pd.read_csv(args2.eval_file, sep='\t')
    qs = df['target_text'].tolist()
    indexes = [i for i in range(len(df))]
    top_k = [1,5,10,20,40,50,100]
    correct_cnt = {k:0 for k in top_k}
    q_embeddings = np.array([get_embedding(q) for q in qs])
    dpr_scores = np.matmul(q_embeddings, embeddings.T)
    for i in range(len(qs)) :
        q_rank = len(embeddings) - ss.rankdata(list(dpr_scores[i]))[indexes[i]] + 1
        for k in top_k :  
            if q_rank <= k :
                correct_cnt[k] += 1
    for k in top_k :
        print ('Top-{} accuracy: {}'.format(k, correct_cnt[k]/len(qs)))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_file', required=True, type=str)
    parser.add_argument('--embeddings', required=True, type=str)
    parser.add_argument('--eval_file', required=True, type=str)
    parser.add_argument('--top_k', required=False, default=[1,5,10,20,40,50,100], type=str)

    args = parser.parse_args()
    main(args)
