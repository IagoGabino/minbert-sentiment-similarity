# sts_regressor.py

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from types import SimpleNamespace
from tqdm import tqdm
import csv
from bert.bert import BertModel
from bert.optimizer import AdamW
from sts_dataset import SentencePairSingleInputDataset
from torch.cuda.amp import GradScaler, autocast

def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def preprocess_string(s):
    return ' '.join(s.lower()
                    .replace('.', ' .')
                    .replace('?', ' ?')
                    .replace(',', ' ,')
                    .replace('\'', ' \'')
                    .split())


def load_sts_data(similarity_filename, split='train'):
    """Carrega dados STS (Semantic Textual Similarity) de um arquivo CSV."""
    similarity_data = []
    
    with open(similarity_filename, 'r', encoding='utf-8') as fp:
        for record in csv.DictReader(fp, delimiter='\t'):
            sent_id = record['id'].strip()
            sentence1 = preprocess_string(record['sentence1'])
            sentence2 = preprocess_string(record['sentence2'])
            
            if split == 'test':
                similarity_data.append((sentence1, sentence2, sent_id))
            else:
                similarity = float(record['similarity'])  # Converte para float
                similarity_data.append((sentence1, sentence2, similarity, sent_id))
    
    print(f"Loaded {len(similarity_data)} {split} examples from {similarity_filename}")
    return similarity_data



class STSRegressor(nn.Module):
    def __init__(self, config):
        super(STSRegressor, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        # Se "pretrain", congela parâmetros; se "finetune", libera parâmetros.
        for param in self.bert.parameters():
            param.requires_grad = (config.option == 'finetune')

        hidden_size = config.hidden_size  # ex: 768
        # Agora, se queremos uma simples regressão, podemos usar: hidden_size -> 1
        self.regression_head = nn.Linear(hidden_size, 1)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def predict_similarity(self, input_ids, attention_mask, token_type_ids=None):
        # forward único: BERT com ambas as sentenças em um input
        out = self.bert(input_ids=input_ids,
                        attention_mask=attention_mask)
        # pooler_output -> [batch_size, hidden_size]
        cls_emb = out['pooler_output']

        # dropout -> regression head
        x = self.dropout(cls_emb)
        logits = self.regression_head(x)  # shape: [batch_size, 1]
        return logits

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Use 'predict_similarity' para pontuar similaridade.")



def train_sts(args):
    device = torch.device('cuda' if args.use_gpu else 'cpu')
    seed_everything(args.seed)

    # Carrega dados
    similarity_train = load_sts_data(args.train, split='train')
    similarity_dev = load_sts_data(args.dev, split='dev')

    # Usa a nova classe SentencePairSingleInputDataset
    train_dataset = SentencePairSingleInputDataset(similarity_train, args, isRegression=True)
    dev_dataset = SentencePairSingleInputDataset(similarity_dev, args, isRegression=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              collate_fn=train_dataset.collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False,
                            collate_fn=dev_dataset.collate_fn)

    # Cria modelo BERT
    config = {
        'hidden_dropout_prob': args.hidden_dropout_prob,
        'hidden_size': 768,
        'option': args.option
    }
    config = SimpleNamespace(**config)

    model = STSRegressor(config).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scaler = GradScaler()
    best_dev_corr = -999

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):

            input_ids = batch['token_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)  # shape [batch_size]

            optimizer.zero_grad()
            with autocast():
                logits = model.predict_similarity(input_ids, attention_mask, token_type_ids).view(-1)
                loss = F.mse_loss(logits, labels, reduction='mean')
            
            # loss.backward()
            # optimizer.step()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        # Avalia no dev
        dev_corr = evaluate_sts(dev_loader, model, device)
        print(f"[Epoch {epoch}] train_loss={total_loss/len(train_loader):.4f} dev_pearson={dev_corr:.4f}")

        if dev_corr > best_dev_corr:
            best_dev_corr = dev_corr
            save_checkpoint(model, optimizer, args, config, args.checkpoint_path)


def evaluate_sts(dataloader, model, device):
    model.eval()
    preds, golds = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['token_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)  # shape [batch_size]

            logits = model.predict_similarity(input_ids, attention_mask, token_type_ids).view(-1)
            preds.extend(logits.cpu().numpy())
            golds.extend(labels.cpu().numpy())

    if len(preds) < 2:
        return 0.0
    return np.corrcoef(preds, golds)[0, 1]



def test_sts(args):
    """
    Avalia o conjunto dev (cálculo de Pearson) e gera previsões para o conjunto de teste.
    As predições são salvas separadamente em arquivos CSV.
    """
    device = torch.device('cuda' if (args.use_gpu and torch.cuda.is_available()) else 'cpu')

    # --- Carregar modelo salvo ---
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    config = SimpleNamespace(**checkpoint['model_config'])
    model = STSRegressor(config).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    # ========================== EVAL NO DEV ==========================
    similarity_dev = load_sts_data(args.dev, split='train')  # split='train' porque tem rótulos
    dev_dataset = SentencePairSingleInputDataset(similarity_dev, args, isRegression=True)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False,
                            collate_fn=dev_dataset.collate_fn)

    dev_preds, dev_labels, dev_sent_ids = [], [], []
    with torch.no_grad():
        for batch in dev_loader:
            input_ids = batch['token_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)  # shape [batch_size]

            logits = model.predict_similarity(input_ids, attention_mask, token_type_ids).view(-1)
            dev_preds.extend(logits.cpu().numpy())
            dev_labels.extend(labels.cpu().numpy())
            dev_sent_ids.extend(batch['sent_ids'])

    # Calcula correlação de Pearson no DEV
    dev_corr = np.corrcoef(dev_preds, dev_labels)[0, 1]
    print(f"[DEV] Pearson Correlation: {dev_corr:.4f}")

    # Salvar predições do DEV para análise posterior
    dev_out_path = args.dev_out
    with open(dev_out_path, 'w', encoding='utf-8') as f:
        f.write("id\tPredicted_Similarity\tTrue_Similarity\n")
        for sid, pred, true in zip(dev_sent_ids, dev_preds, dev_labels):
            f.write(f"{sid}\t{pred:.4f}\t{true}\n")
    print(f"Predições do conjunto DEV salvas em {dev_out_path}")

    # ========================== INFERÊNCIA NO TESTE ==========================
    similarity_test = load_sts_data(args.test, split='test')  # split='test' porque não tem rótulos
    test_dataset = SentencePairSingleInputDataset(similarity_test, args, isRegression=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             collate_fn=test_dataset.collate_fn)

    test_preds, test_sent_ids = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['token_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)

            logits = model.predict_similarity(input_ids, attention_mask, token_type_ids).view(-1)
            test_preds.extend(logits.cpu().numpy())
            test_sent_ids.extend(batch['sent_ids'])

    # Salvar predições do TESTE para envio/futuras análises
    test_out_path = args.test_out
    with open(test_out_path, 'w', encoding='utf-8') as f:
        f.write("id\tPredicted_Similarity\n")
        for sid, pred in zip(test_sent_ids, test_preds):
            f.write(f"{sid}\t{pred:.4f}\n")
    print(f"Predições do conjunto TESTE salvas em {test_out_path}")



def save_checkpoint(model, optimizer, args, config, path):
    save_data = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'model_config': vars(config)
    }
    torch.save(save_data, path)
    print(f"Checkpoint salvo em {path}")
