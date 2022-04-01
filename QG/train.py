import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BartTokenizer, BartForConditionalGeneration, AutoModelForSeq2SeqLM
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


def train(args, epoch, tokenizer, model, device, loader, optimizer):
    model.train()
    for _,data in enumerate(loader, 0):
        y = data['target_ids'].to(device, dtype = torch.long)
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone().detach()
        lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
        ids = data['source_ids'].to(device, dtype = torch.long)
        mask = data['source_mask'].to(device, dtype = torch.long)

        outputs = model(input_ids = ids, attention_mask = mask, decoder_input_ids=y_ids, lm_labels=lm_labels)
        loss = outputs[0]

        if _%30 == 0 :
            print ('Epoch {}: Completed {} batches out of {}'.format(epoch+1, _, len(loader)))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def main(args):

    torch.manual_seed(42) # pytorch random seed
    np.random.seed(42) # numpy random seed
    torch.backends.cudnn.deterministic = True

    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    
    train_df = pd.read_csv(args.train_file, sep='\t')
        
    print("TRAIN Dataset: {}".format(train_df.shape))

    # Creating the Training and Validation dataset for further creation of Dataloader
    training_set = CustomDataset(train_df, tokenizer, 512, 150)

    # Defining the parameters for creation of dataloaders
    train_params = {
        'batch_size': args.batch_size,
        'shuffle': True,
        'num_workers': 2
    }

    training_loader = DataLoader(training_set, **train_params)

    model = AutoModelForSeq2SeqLM.from_pretrained("McGill-NLP/bart-qg-nq-checkpoint")
    model = model.to(device)
    optimizer = torch.optim.Adam(params =  model.parameters(), lr=args.lr)

    print('Starting QG model training...')

    for epoch in range(args.epochs):
        train(args, epoch, tokenizer, model, device, training_loader, optimizer)
        print(f'Finished Epoch: {epoch+1}')
        print ('---------------------------------------------------------------------')
    model.save_pretrained('outputs/')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', required=False, type=int, default=1)
    parser.add_argument('--batch_size', required=False, type=int, default=32)
    parser.add_argument('--lr', required=False, type=float, default=1e-5)
    parser.add_argument('--train_file', required=False, type=str, default='data/train.tsv')
    parser.add_argument('--checkpoint', required=False, type=str, default='facebook/bart-base')
    args = parser.parse_args()
    main(args)
