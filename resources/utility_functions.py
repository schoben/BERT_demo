"""This file contains helper functions used by both amazon and conll003 scripts"""

import numpy as np
from datasets import load_metric
from sklearn.model_selection import train_test_split
from transformers import BertTokenizerFast, DistilBertTokenizerFast, EvalPrediction


BERT_TYPES = type(DistilBertTokenizerFast) or type(BertTokenizerFast)


def split_data(data: list, labels: list, random_state: int = 42, train_size: int = 0.6) -> \
        (list, list, list, list, list, list):
    """This function creates training, test, and validation sets if needed
    :param data - the array fo data to be shuffled
    :param labels - the labels for the data
    :param random_state - the random state to use if desired, otherwise defaults to 42
    :param train_size - the proportion of data to be used in the training set
    :return train_set, test_set, validation_set - 3 arrays are returned for training, testing, and validiation"""
    # First create training set
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, random_state=random_state,
                                                                        test_size=1 - train_size)
    # Then split test set to get validation set
    test_data, validation_data, test_Labels, validation_labels = train_test_split(test_data, test_labels,
                                                                                  random_state=random_state, test_size=0.5)
    return train_data, train_labels, test_data, test_labels, validation_data, validation_labels


def seq_eval(eval_pred: EvalPrediction) -> dict:
    """This is a function to compute accuracy with regard to the validation set
    :param eval_pred - the predicted values
    :return metric - a dict containing the calculated metric"""
    # Start by loading the metric to be used.
    accuracy = load_metric('accuracy')

    #Calculate said metric and output
    preds, labels = eval_pred
    predictions = np.argmax(preds, axis=-1)
    metric = accuracy.compute(predictions=predictions, references=labels)
    # Print and return the values to the trainer object
    print(metric)
    return metric


def ner_eval(eval_pred: EvalPrediction) -> dict:
    """This is a function that calculates accuracy for token classification
    :param eval_pred - an object containing predictions and labels for evaluation
    :return results_dict - a dict containing the results of the computation"""
    # Load the metric to be used.
    metric = load_metric('seqeval')
    # Get the prediction data
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)
    # This map was created to reconvert the numerical labels back to their original
    # ner tokens so the trainer can compute accuracy
    id_mapping = {0: 'I-ORG', 1: 'I-PER', 2: 'I-MISC', 3: 'B-PER',
     4: 'O', 5: 'I-LOC', 6: 'B-MISC', 7: 'B-ORG', 8: 'B-LOC'}

    # We need to take out data labeled -100 as it is ignored and not part of the model
    # These loops separate out the ignored tokens from the ones with labels
    # and place them in lists directly above them.
    true_predictions = []
    true_labels = []
    for prediction, label in zip(predictions, labels):
        true_pred = [id_mapping[pred] for pred, lab in zip(prediction, label) if lab != -100]
        true_lab = [id_mapping[lab] for pred, lab in zip(prediction, label) if lab != -100]
        true_predictions.append(true_pred)
        true_labels.append(true_lab)

    # Calcualte the results
    results = metric.compute(predictions=true_predictions, references=true_labels)
    return_dict = {
        "accuracy": results["overall_accuracy"]
    }
    # Print and return results
    print(return_dict)
    return return_dict


