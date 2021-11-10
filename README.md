# BERT_demo
Created by: Benjamin Scholom

This is a repository which takes a small amount of data from two data sets: 
one contains sentences from the CoNLL003 set where words are labeled with named entities 
while the other consists of amazon reviews along with their ratings and other metadata.

The purpose of this repository is to convert the raw data from each set into
a format that can be read by a BERT model than then used to fine tune said model.
In this case, we are using a Distilled Bert implementation for both the sequence classification 
needed by the Amazon review set and for the token classification needed by the CoNLL003 set.

To get started go into the main repository directory run:

```pip install -r requirements.txt```

After that:

```python run_all_models.py```

Thank you for viewing this demo!

