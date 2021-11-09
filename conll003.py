# -*- coding: utf-8 -*-
"""This file loads data from the conll003 dataset and uses a fine-tuned BERT model for named entity recognition"""

import zipfile as zp
import numpy as np
from resources.utility_functions import split_data
from resources.train_model import train_model
from resources.bert_dataset import BertData
from transformers import DistilBertTokenizerFast


# The base path where the data archive is located.
PATH_TO_DATA: str = './data/archive.zip'

# The file suffixes with data.
FILES: dict = {'train': 'train.txt',
               'test': 'test.txt',
               'valid': 'valid.txt'}

# The relevant fields to collect.
# We want only the token and the named entity for this demonstration.
FIELDS: dict = {'token': 0, 'ne': 3}

# We want tokens for capitalized words now that we are working with named entities
PRETRAINED = 'distilbert-base-cased'


def read_data(path: str, file: str) -> (list, list):
    """This function opens the zip archive with the data and retrieves data from a file inside
    The function intends to save memory by reading from a zip archive and iterating through each row
    :param path - the str path where the archive is located
    :param file - the the file name with the appropriate data - could be train, test, or valid
    depending on which data set we are looking at for the given task
    :return data, labels - two lists, one containing all the rows data and another the labels"""
    # initialize data list to store completed sentences
    data = []
    labels = []
    # this list collects sentences as they are being parsed
    sentence = []
    sentence_labels = []
    # open archive
    with zp.ZipFile(path) as archive:
        # open file containing data within archive
        with archive.open(file) as f:
            # iterate over rows (lines)
            for row in f:
                if type(row) == bytes:
                    row = row.decode('utf8')
                # these values indicate a sentence ending that should be added to the data
                if row == '-DOCSTART- -X- -X- O\n' or row == '\n':
                    if len(sentence) > 0:
                        # if there is a complete sentence present in the sentence list add to data
                        data.append(sentence)
                        labels.append(sentence_labels)
                        # reset sentence list
                        sentence = []
                        sentence_labels = []
                # without the newline breaks, the row contains sentence parts that need to be parsed
                else:
                    # split to easily parse elements of the row
                    r = row.split(' ')
                    # add token to and named entity data to respective sentence fields
                    sentence.append(r[FIELDS['token']])
                    sentence_labels.append(r[FIELDS['ne']].strip('\n'))
    # return two separate lists for easier
    return data, labels


def select_subtoken(tags: list, offset_mapping: list, mapping: dict):
    """function to ensure that each label only has one associated token by
    by selecting the first subtoken for tokens with multiple subtokens.
    This is done by encoding labels so the model ignores certain subtokens.
    :param labels: the list of labels
    :param offset_mapping: the list off positions of subtokens for each sentence in the dataset
    start and end position of the subtokens for every token
    :param mapping - a dict that contains named entity classes that converts then to int ids
    :return encoded_labels - an list of labels suitable for use with the list of tokens in the BERT model"""
    # We will leverage the previously made target to int map for get numerical values for each target (label) variab
    labels = [[mapping[tag] for tag in sent] for sent in tags]
    encoded_labels = []
    for row_labels, row_offset in zip(labels, offset_mapping):
        # We are creating an array corresponding the len off the offset map of -100
        # because this is the number used by the model to determine which tokens to ignore.
        row_encode = np.ones(len(row_offset), dtype=int) * -100
        offset_array = np.array(row_offset)

        # We will select only the subtokens in the first position relative to their main tokens to
        # preserve the the consistent shape of the label and feature tensors.
        # All subtokens in the second or third position relative to their main token will be ignored.
        # When first offset for a label is 0 and the second is not, this is the first label
        # for the first subtoken and we want to keep it. Otherwise, we will ignore
        row_encode[(offset_array[:, 0] == 0) & (offset_array[:, 1] != 0)] = row_labels
        encoded_labels.append(row_encode.tolist())
    return encoded_labels

def main():
    """The main method of the module. This function runs the code that will load, tokenize, and train."""
    raw_data, raw_labels = read_data(PATH_TO_DATA, FILES['train'])
    # split the data into train and valdiation sets
    train_data, train_label, test_data, test_label, val_data, val_label = split_data(data=raw_data[0:2000], labels=raw_labels[0:2000])
    # tokenize the text, but ensure that the tokenizer understands that words are already separated
    #tokenizer = get_tokenizer(name='distilbert', case='cased')
    tokenizer = DistilBertTokenizerFast.from_pretrained(PRETRAINED)
    tokens_train = tokenizer(train_data, truncation=True, padding=True,
                             is_split_into_words=True, return_offsets_mapping=True)
    tokens_val = tokenizer(val_data, truncation=True, padding=True,
                           is_split_into_words=True, return_offsets_mapping=True)
    # since we are tokenizing only one token at a time, we need to account for words that the BERT model may not have
    # BERT will split unknown tokens into known sub-tokens, but since we are doing token classification,
    # we need to have only one token for each label, so we will choose only 1 of the sub-tokens
    # we need to start by getting a list of the total unique labels for the tokens
    target_classes = set(tag for row in raw_labels for tag in row)
    # create a 2 way mapping for these classes
    tag_ids = {tag: i for i, tag in enumerate(target_classes)}
    # We are ensuring that only the first subtoken out of a split token gets a label
    train_label = select_subtoken(tags=train_label, offset_mapping=tokens_train['offset_mapping'], mapping=tag_ids)
    val_label = select_subtoken(tags=val_label, offset_mapping=tokens_val['offset_mapping'], mapping=tag_ids)

    # remove offset mapping before creating datasets for pytorch
    tokens_train.pop('offset_mapping')
    tokens_val.pop('offset_mapping')
    ner_set = BertData(tokens=tokens_train, labels=train_label)
    val_set = BertData(tokens=tokens_val, labels=val_label)
    train_model(data=ner_set, val=val_set, num_labels=len(target_classes), seq=False)
    print('done')

if __name__ == "__main__":
    main()