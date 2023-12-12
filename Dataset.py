import spacy
from tqdm import tqdm
from torch.utils.data import Dataset
import numpy as np
from nltk import FreqDist
import pickle
import torch

class CustomDataset(Dataset):
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        f = open(dataset_path + 'train_korean.txt', 'r')
        self.ko_dataset = f.readlines()
        f.close()
        f = open(dataset_path + 'train_english.txt', 'r')
        self.en_dataset = f.readlines()
        f.close()
        self.ko_model = spacy.load("ko_core_news_sm")
        self.en_model = spacy.load("en_core_web_sm")
        
        with open('ko_pos_to_index.pickle', 'rb') as fr:
            self.ko_pos_to_index = pickle.load(fr)
        with open('en_pos_to_index.pickle', 'rb') as fr:
            self.en_pos_to_index = pickle.load(fr)

    def __len__(self):
        return len(self.ko_dataset)

    def __getitem__(self, idx):
        ko_line = self.ko_dataset[idx]
        en_line = self.en_dataset[idx]
        ko_processing = self.ko_model(ko_line)
        en_processing = self.en_model(en_line)
        
        ko_tokens = []
        en_tokens = ['sos']
        
        for token in ko_processing:
            ko_tokens.append(self.ko_pos_to_index[token.pos_])
        ko_tokens.append('eos')
            
        for token in en_processing:
            en_tokens.append(self.en_pos_to_index[token.pos_])
            
        return ko_line, en_line, ko_tokens, en_tokens
    
    def collate_fn(self, batch):
        
        ko_lines, en_lines, ko_tokens, en_tokens = zip(*batch)
        batch_size = len(en_lines)

        # Merge (from tuple of 1D tensor to 2D tensor).
        en_token_lengths = [len(text) for text in en_lines]
        en_max_token_length = max(en_token_lengths)
          
        ko_token_lengths = [len(text) for text in ko_lines]
        ko_max_token_length = max(ko_token_lengths)
        
        en_lines_pads, en_tokens_pads = [torch.zeros(batch_size, en_max_token_length, dtype=torch.long) for _ in range(2)]
        ko_lines_pads, ko_tokens_pads = [torch.zeros(batch_size, ko_max_token_length, dtype=torch.long) for _ in range(2)]
     
        for idx in range(batch_size):
            ko_token_end = ko_token_lengths[idx]
            en_token_end = en_token_lengths[idx]
            
            ko_lines_pads[idx, :ko_token_end] = ko_lines[idx]        
            en_lines_pads[idx, :en_token_end] = en_lines[idx]   
            
            ko_tokens_pads[idx, :ko_token_end] = ko_tokens[idx]
            en_tokens_pads[idx, :en_token_end] = en_tokens[idx]     
        
        return ko_lines_pads, en_lines_pads, ko_tokens_pads, en_tokens_pads
        
# ko_dataset_path = '/home/ywkim/NLP/train_korean.txt'
# en_dataset_path = '/home/ywkim/NLP/train_english.txt'
