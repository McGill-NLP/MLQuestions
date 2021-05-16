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


def main(args) :
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    namespace = Namespace(batch_size=32, distributed_world_size=1, device=device, do_lower_case=False, encoder_model_type='hf_bert', 
                     fp16=False, fp16_opt_level='O1', local_rank=-1, 
                     model_file=args.model_file, 
                     n_gpu=1, no_cuda=False, num_shards=1, pretrained_file=None, 
                     pretrained_model_cfg='bert-base-uncased', projection_dim=0, sequence_length=512, shard_id=0)
    saved_state = load_states_from_checkpoint(namespace.model_file)
    set_encoder_params_from_state(saved_state.encoder_params, namespace)
    tensorizer, encoder, _ = init_biencoder_components(namespace.encoder_model_type, namespace, inference_only=True)
    encoder = encoder.question_model
    encoder, _ = setup_for_distributed_mode(encoder, None, namespace.device, namespace.n_gpu,
                                            namespace.local_rank,
                                            namespace.fp16)
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

    UP = pd.read_csv('../data/passages_unaligned.tsv', sep='\t')['input_text'].tolist()
    with open(args.embeddings, 'rb') as f:
        embeddings = pickle.load(f)
    UQ = pd.read_csv('../data/questions_unaligned.tsv', sep='\t')['target_text'].tolist()
    df_pseudo = pd.DataFrame()
    psuedo_questions, pseudo_paras, psuedo_scores = UQ, [], []
    score_sum = 0
    print ('starting')
    embeddings = np.array([e[1] for e in embeddings])
    q_embeddings = np.array([get_embedding(q) for q in UQ])
    dpr_scores = np.matmul(q_embeddings, embeddings.T)
    for i in range(len(UQ)) :
        scores = list(dpr_scores[i])
        max_score = max(scores)
        score_sum += max_score
        pseudo_paras.append(UP[scores.index(max_score)])
        psuedo_scores.append(max_score)
        if i%500==0 :
            print ('Completed {} out of {} questions. Moving average DPR score = {}'.format(i+1, len(UQ), score_sum/(i+1)))
    f.close()

    df_pseudo['input_text'] = pd.Series(pseudo_paras)
    df_pseudo['target_text'] = pd.Series(psuedo_questions)
    df_pseudo.to_csv(args.out_file, sep='\t')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_file', required=True, type=str)
    parser.add_argument('--embeddings', required=True, type=str)
    parser.add_argument('--out_file', required=True, type=str)
    args = parser.parse_args()
    main(args)