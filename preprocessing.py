import spacy
from tqdm import tqdm
from torch.utils.data import Dataset
import numpy as np
from nltk import FreqDist
import pickle

dataset_path = '/home/ywkim/NLP/'
f = open(dataset_path + 'train_korean.txt', 'r')
ko_dataset = f.readlines()
f.close()
f = open(dataset_path + 'train_english.txt', 'r')
en_dataset = f.readlines()
f.close()
ko_model = spacy.load("ko_core_news_sm")
en_model = spacy.load("en_core_web_sm")

ko_tokenized = []
for sentence in tqdm(ko_dataset):
    temp = ko_model(sentence)
    for token in temp:
        ko_tokenized.append(token.pos_)
ko_pos_vocab = FreqDist(np.hstack(ko_tokenized))
print('단어 집합의 크기 : {}'.format(len(ko_pos_vocab)))

ko_pos_to_index = {word[0] : index + 1 for index, word in enumerate(ko_pos_vocab)}
with open('ko_pos_to_index.pickle','wb') as fw:
    pickle.dump(ko_pos_to_index, fw)

en_tokenized = []
for sentence in tqdm(en_dataset):
    temp = en_model(sentence)
    for token in temp:
        en_tokenized.append(token.pos_)
en_pos_vocab = FreqDist(np.hstack(en_tokenized))
print('단어 집합의 크기 : {}'.format(len(en_pos_vocab)))

en_pos_to_index = {word[0] : index + 1 for index, word in enumerate(en_pos_vocab)}
with open('en_pos_to_index.pickle','wb') as fw:
    pickle.dump(en_pos_to_index, fw)
