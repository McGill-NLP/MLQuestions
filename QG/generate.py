import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BartTokenizer, BartForConditionalGeneration
from torch import cuda

device = 'cuda' if cuda.is_available() else 'cpu'

class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer, source_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.ctext = self.data.input_text

    def __len__(self):
        return len(self.ctext)

    def __getitem__(self, index):
        ctext = str(self.ctext[index])
        ctext = ' '.join(ctext.split())

        source = self.tokenizer.batch_encode_plus([ctext], max_length= self.source_len, pad_to_max_length=True,return_tensors='pt', truncation='longest_first')

        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()

        return {
            'source_ids': source_ids.to(dtype=torch.long), 
            'source_mask': source_mask.to(dtype=torch.long)
        }

def validate(tokenizer, model, device, loader):
    model.eval()
    predictions = []
    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            ids = data['source_ids'].to(device, dtype = torch.long)
            mask = data['source_mask'].to(device, dtype = torch.long)

            generated_ids = model.generate(
                input_ids = ids,
                attention_mask = mask, 
                max_length=512, 
                do_sample=True,
                top_k=50
            )
            
            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
            predictions.extend(preds)
    return predictions

def main(args):
    
    torch.manual_seed(42) # pytorch random seed
    np.random.seed(42) # numpy random seed
    torch.backends.cudnn.deterministic = True

    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    
    df = pd.read_csv(args.file, sep='\t')
    df_set = CustomDataset(df, tokenizer, 512)
    df_params = {
        'batch_size': 32,
        'shuffle': False,
        'num_workers': 2
        }
    df_loader = DataLoader(df_set, **df_params)
    
    model = BartForConditionalGeneration.from_pretrained(args.checkpoint)
    model = model.to(device)
    predictions = validate(tokenizer, model, device, df_loader)
    
    final_df = pd.DataFrame()
    final_df['input_text'] = df['input_text']
    final_df['target_text'] = pd.Series(predictions)
    final_df.to_csv(args.checkpoint+'QG-predictions-50K.tsv', sep='\t')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=False, type=str, default='all')
    parser.add_argument('--file', required=False, type=str, default='data/unlabeled_passages.tsv')
    args = parser.parse_args()
    main(args)