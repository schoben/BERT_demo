"""This file contains the pytorch dataset class for use by both models to contain the labels
 and encodings of the data"""

import torch
from torch.utils.data import Dataset


class BertData(Dataset):
    """a pytorch dataset class that takes tokenized BERT data and labels and prepares it for use
    in training"""
    def __init__(self, tokens: list, labels: list):
        """initialization function for the object
        :param tokens - the text data that has been encoded with a BERT tokenizer
        :param labels - a list of the associated labels for this data"""
        self.tokens = tokens
        self.labels = labels

    def __getitem__(self, index: int) -> dict:
        """get item function for use by pytorch to pull specific samples from the dataset
        :param index - the index value of a given sample
        :return item - a dict containing the input id tensor, the attention mask tensor,
        and the label tensor. All are needed for input into a bert model"""
        # create a dictionary out of sequences or tokens and their attention masks
        item = {key: torch.tensor(val[index]) for key, val in self.tokens.items()}
        # add label tensor to the dictionary
        item['label'] = torch.tensor((self.labels[index]))
        return item

    def __len__(self):
        return len(self.labels)
