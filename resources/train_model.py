"""file to train a BERT model of a custom dataset"""


import torch
from torch.utils.data import DataLoader
from transformers import DistilBertForSequenceClassification, DistilBertForTokenClassification, AdamW



def train_model(data:object, num_labels: int, seq: bool):
    """this function trains  a fine tuned bert model off a custom dataset"""
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # if training on sequences uses sequences model, otherwise use tokens model
    if seq:
        model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_labels)
    else:
        model = DistilBertForTokenClassification.from_pretrained('distilbert-base-cased', num_labels=num_labels)
    model.to(device)
    model.train()

    train_loader = DataLoader(data, batch_size=32, shuffle=True)

    optim = AdamW(model.parameters(), lr=5e-5)

    for epoch in range(3):
        for batch in train_loader:
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            loss.backward()
            optim.step()

    model.eval()