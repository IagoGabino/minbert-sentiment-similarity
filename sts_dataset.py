#!/usr/bin/env python3

'''
This module contains our Dataset classes and functions that load the three datasets
for training and evaluating multitask BERT.

Feel free to edit code in this file if you wish to modify the way in which the data
examples are preprocessed.
'''

import csv

import torch
from torch.utils.data import Dataset
from bert.tokenizer import BertTokenizer


# datasets.py

class SentencePairSingleInputDataset(Dataset):
    def __init__(self, dataset, args, isRegression=False):
        """
        dataset: lista de tuplas (sent1, sent2, label, id)  ou (sent1, sent2, id) no teste
        isRegression: True se for STS (similaridade real), False se for classificação binária
        """
        self.dataset = dataset
        self.p = args
        self.isRegression = isRegression
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def pad_data(self, data):
        """
        data = [ (sent1, sent2, label, sent_id), ... ]  (treino/dev)
         ou    [ (sent1, sent2, sent_id), ... ]        (teste)
        """
        # Verifica se há label ou não
        has_label = (len(data[0]) == 4)

        sent1 = [x[0] for x in data]
        sent2 = [x[1] for x in data]
        sent_ids = [x[-1] for x in data]  # sempre a última posição

        if has_label:
            labels = [x[2] for x in data]
        else:
            labels = None

        # A chamada abaixo faz o "par completo": [CLS] sent1 [SEP] sent2 [SEP]
        encoding = self.tokenizer(
            sent1,
            sent2,
            return_tensors='pt',
            padding=True,
            truncation=True
        )
        # A saída do tokenizer tem input_ids, attention_mask, token_type_ids
        token_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        token_type_ids = encoding['token_type_ids']

        if has_label:
            if self.isRegression:
                labels = torch.tensor(labels, dtype=torch.float)
            else:
                labels = torch.tensor(labels, dtype=torch.long)
        else:
            labels = None

        return token_ids, attention_mask, token_type_ids, labels, sent_ids

    def collate_fn(self, batch_data):
        token_ids, attention_mask, token_type_ids, labels, sent_ids = self.pad_data(batch_data)
        batched_data = {
            'token_ids': token_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'labels': labels,
            'sent_ids': sent_ids
        }
        return batched_data
