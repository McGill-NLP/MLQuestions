import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BartTokenizer, BartForConditionalGeneration
from torch import cuda
from nlgeval import compute_metrics
import os

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


def validate(tokenizer, model, device, loader):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            y = data['target_text']
            ids = data['source_ids'].to(device, dtype = torch.long)
            mask = data['source_mask'].to(device, dtype = torch.long)

            generated_ids = model.generate(
                input_ids = ids,
                attention_mask = mask,
                do_sample=True,
                max_length=512,
                top_k=50
                )
            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
            target = [t for t in y]

            predictions.extend(preds)
            actuals.extend(target)
    return predictions, actuals

def main(args):

    torch.manual_seed(42) # pytorch random seed
    np.random.seed(42) # numpy random seed
    torch.backends.cudnn.deterministic = True

    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    
    val_params = {
        'batch_size': 32,
        'shuffle': False,
        'num_workers': 2
    }
    
    model = BartForConditionalGeneration.from_pretrained(args.checkpoint)
    model = model.to(device)
    optimizer = torch.optim.Adam(params =  model.parameters(), lr=1e-5)
    
    val_df = pd.read_csv(args.eval_file, sep='\t')
    val_set = CustomDataset(val_df, tokenizer, 512, 150)
    val_loader = DataLoader(val_set, **val_params)
    predictions, actuals = validate(tokenizer, model, device, val_loader)
    print_scores(actuals, predictions)
    
def print_scores(ref, pred) :
    f = open('ref.txt', 'w')
    for r in ref :
        f.write(r+'\n')
    f.close()
    f = open('pred.txt', 'w')
    for p in pred :
        f.write(p+'\n')
    f.close()
    try :
        compute_metrics(hypothesis='pred.txt',
                               references=['ref.txt'])
    except:
        pass
    #delete intermediate files.
    os.remove('ref.txt')
    os.remove('pred.txt')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, type=str)
    parser.add_argument('--eval_file', required=False, type=str, default=None)
    args = parser.parse_args()
    main(args)