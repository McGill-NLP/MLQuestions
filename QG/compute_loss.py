import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BartTokenizer, BartForConditionalGeneration
from torch import cuda

device = 'cuda' if cuda.is_available() else 'cpu'

class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer, source_len, target_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.target_len = target_len
        self.text = self.data.target_text
        self.ctext = self.data.input_text

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        ctext = str(self.ctext[index])
        ctext = ' '.join(ctext.split())

        text = str(self.text[index])
        text = ' '.join(text.split())

        source = self.tokenizer.batch_encode_plus([ctext], max_length= self.source_len, pad_to_max_length=True,return_tensors='pt', truncation='longest_first')
        target = self.tokenizer.batch_encode_plus([text], max_length= self.target_len, pad_to_max_length=True,return_tensors='pt', truncation='longest_first')

        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        target_ids = target['input_ids'].squeeze()
        target_mask = target['attention_mask'].squeeze()

        return {
            'source_ids': source_ids.to(dtype=torch.long), 
            'source_mask': source_mask.to(dtype=torch.long), 
            'target_ids': target_ids.to(dtype=torch.long),
            'target_ids_y': target_ids.to(dtype=torch.long),
            'target_text': text
        }

def compute_losses(tokenizer, model, device, loader):
    model.train()
    losses = []
    for _,data in enumerate(loader, 0):
        y = data['target_ids'].to(device, dtype = torch.long)
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone().detach()
        lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
        ids = data['source_ids'].to(device, dtype = torch.long)
        mask = data['source_mask'].to(device, dtype = torch.long)

        outputs = model(input_ids = ids, attention_mask = mask, decoder_input_ids=y_ids, lm_labels=lm_labels)
        losses.append(str(outputs[0].item()))
        
        if _%1000==0 : 
            print ('Completed {} out of {}'.format(_, len(loader))) 
    return losses

def main(args) :
    
    torch.manual_seed(42) # pytorch random seed
    np.random.seed(42) # numpy random seed
    torch.backends.cudnn.deterministic = True
    
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    
    train_df = pd.read_csv(args.file, sep='\t')
    if 'target_text' not in train_df :
        train_df['target_text'] = train_df['target_text0']
    train_df = train_df[['input_text','target_text']]
    training_set = CustomDataset(train_df, tokenizer, 512, 150)
    train_params = {
        'batch_size': 1,
        'shuffle': True,
        'num_workers': 2
        }
    training_loader = DataLoader(training_set, **train_params)
    model = BartForConditionalGeneration.from_pretrained(args.checkpoint)
    model = model.to(device)
    
    losses = compute_losses(tokenizer, model, device, training_loader)
    f = open(args.save_to, 'w')
    f.write('\n'.join(losses))
    f.close()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', required=False, type=str, default='data/dev_1.tsv')
    parser.add_argument('--checkpoint', required=True, type=str)
    parser.add_argument('--save_to', required=True, type=str)
    args = parser.parse_args()
    main(args)