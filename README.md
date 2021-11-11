# BERT_demo
Created by: Benjamin Scholom

This is a repository which takes a small amount of data from two data sets: 
one contains sentences from the CoNLL003 set where words are labeled with named entities 
while the other consists of amazon reviews along with their ratings and other metadata.

The purpose of this repository is to convert the raw data from each set into
a format that can be read by a BERT model than then used to fine tune said model.
In this case, we are using a Distilled Bert implementation for both the sequence classification 
needed by the Amazon review set and for the token classification needed by the CoNLL003 set.

After tokenization, the both datasets have feature tensors containing rows 
of examples with each example containing the same number of tokens. 
Some of these tokens are padding to ensure all sequences are the same length. 
However, the label tensors are not the same, with the Amazon dataset having 1 dimensional
data with a rating value for every sequence. The CoNLL003 dataset has 2 dimensional labels - 
the label corresponding to each sequence is also a sequence containing a token representing either a label class
or a code indicating that the model should skip the token. Given that the Amazon dataset is trying
to predict some form of sentiment (ratings) on whole sequences 
while the CoNLL003 set is being used to identify types of named entity on a word basis,
the difference in shape is not surprising.

The requirements are listed in requirements.txt. If needed, they can be installed. Make sure you are using python 3.6-3.9 before running this repository.
Go into the main repository directory (BERT_demo) and run:

```pip install -r requirements.txt```

After that you can run all the models:

```python run_models.py```

Or just run the 1 of them:

```python run_models.py -n``` for Named Entity Recognition on the CoNLL003 data.

```python run_models.py -a``` for Sentiment Classification on the Amazon review set.

Thank you for viewing this demo!

