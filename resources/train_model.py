"""file to train a BERT model of a custom dataset"""


import torch
from torch.utils.data import DataLoader
import numpy as np
import time
from transformers import DistilBertForSequenceClassification, DistilBertForTokenClassification, AdamW, \
    DistilBertTokenizerFast, BertTokenizerFast, TrainingArguments, BertForSequenceClassification, Trainer, TrainingArguments
from resources.utility_functions import seq_eval, ner_eval, BERT_TYPES
from datasets import load_metric
from accelerate import Accelerator





def train_model(data: BERT_TYPES, val: BERT_TYPES, num_labels: int, seq: bool):
    """this function trains  a fine tuned bert model off a custom dataset
    :param data - the tokenized data to be trained
    :param val - the teokenized data for validation
    :param num_labels - the number of unique label values
    :param seq - true if doing sequence classification false for token classification"""
    #tic = time.perf_counter()
    # accelerator = Accelerator()
    # device = accelerator.device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # if training on sequences uses sequences model, otherwise use tokens model
    if seq:
        model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_labels)
        eval_func = seq_eval
    else:
        model = DistilBertForTokenClassification.from_pretrained('distilbert-base-cased', num_labels=num_labels)
        eval_func = ner_eval
    # use device that has been selected - hopefully a gpu
    model.to(device)

    # ensure that training data is shuffled
    # train_loader = DataLoader(data, batch_size=16, shuffle=True)
    # val_loader = DataLoader(val, batch_size=32)
    # optim = AdamW(model.parameters(), lr=5e-5)
    #
    # #model, optim, accelerator.prepare(model, optim, train_loader, val_loader)
    # model.train()
    #
    # for epoch in range(3):
    #
    #     for batch in train_loader:
    #         optim.zero_grad()
    #         input_ids = batch['input_ids'].to(device)
    #         attention_mask = batch['attention_mask'].to(device)
    #         labels = batch['labels'].to(device)
    #         outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
    #         loss = outputs.loss
    #         loss.backward()
    #         optim.step()
    #
    #     model.eval()
    #     metric = load_metric("accuracy")
    #
    #     for batch in val_loader:
    #         batch = {keys: values.to(device) for keys, values in batch.items()}
    #         with torch.no_grad():
    #             outputs = model(**batch)
    #
    #         logits = outputs.logits
    #         predictions = torch.argmax(logits, dim=-1)
    #         metric.add_batch(predictions=predictions, references=batch["labels"])
    #
    #     print(metric.compute())
        # for batch in val_loader:
        #     outputs = model(**batch)
        #     predictions = outputs.logits.argmax(dim=-1)
        #     metric.add_batch(
        #         predictions=accelerator.gather(predictions),
        #         references=accelerator.gather(batch["labels"]),
        #     )
        #
        # eval_metric = metric.compute()
        # print(f"epoch {epoch}: {eval_metric}")
    # toc = time.perf_counter()
    # print(toc - tic)

    training_args = TrainingArguments(
        output_dir='results',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_steps=100,
    )


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=data,
        eval_dataset=val,
        compute_metrics=eval_func,
    )

    trainer.train()
    trainer.evaluate()