"""file to train a BERT model of a custom dataset"""


import numpy as np
from transformers import DistilBertForSequenceClassification, DistilBertForTokenClassification,  \
    DistilBertTokenizerFast, Trainer, TrainingArguments, EvalPrediction
from datasets import load_metric

# This is being used to assist with annotation.
BERT_TYPES = type(DistilBertTokenizerFast)
# This constant is the label assigned to tokens ignored in NER classification.


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
    mapping = {0: 'I-ORG', 1: 'I-PER', 2: 'I-MISC', 3: 'B-PER',
               4: 'O', 5: 'I-LOC', 6: 'B-MISC', 7: 'B-ORG', 8: 'B-LOC'}

    # We need to take out data labeled -100 as it is ignored and not part of the model
    # These loops separate out the ignored tokens from the ones with labels
    # and place them in lists directly above them.
    true_predictions = []
    true_labels = []
    for prediction, label in zip(predictions, labels):
        true_pred = [mapping[pred] for pred, lab in zip(prediction, label) if lab != -100]
        true_lab = [mapping[lab] for pred, lab in zip(prediction, label) if lab != -100]
        true_predictions.append(true_pred)
        true_labels.append(true_lab)

    # Calculate the results
    results = metric.compute(predictions=true_predictions, references=true_labels)
    return_dict = {
        "accuracy": results["overall_accuracy"]
    }
    # Print and return results
    print(return_dict)
    return return_dict


def seq_eval(eval_pred: EvalPrediction) -> dict:
    """This is a function to compute accuracy with regard to the validation set
    :param eval_pred - the predicted values
    :return metric - a dict containing the calculated metric"""
    # Start by loading the metric to be used.
    accuracy = load_metric('accuracy')

    # Calculate said metric and output
    preds, labels = eval_pred
    predictions = np.argmax(preds, axis=-1)
    metric = accuracy.compute(predictions=predictions, references=labels)
    # Print and return the values to the trainer object
    print(metric)
    return metric


def train_model(data: BERT_TYPES, val: BERT_TYPES, num_labels: int, seq: bool):
    """this function trains  a fine tuned bert model off a custom dataset
    :param data - the tokenized data to be trained
    :param val - the tokenized data for validation
    :param num_labels - the number of unique label values
    :param seq - true if doing sequence classification false for token classification"""
    # if training on sequences uses sequences model, otherwise use tokens model
    if seq:
        model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_labels)
        eval_func = seq_eval
    else:
        model = DistilBertForTokenClassification.from_pretrained('distilbert-base-cased', num_labels=num_labels)
        eval_func = ner_eval

    # Use trainer object for easy and optimized training
    # specify parameters using the Training arguments
    # Since this is a demo, most values will be set arbitrarily
    training_args = TrainingArguments(
        # not necessary for the demo, but a required parameter
        output_dir='results',
        # This is number of passes desired
        # Normally, we would do 3 or more, but this a demo
        num_train_epochs=1,
        # These are the size of the batches
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        # This is warm up steps to help model adjust to data
        warmup_steps=20,
        # This is the weight decay for tine tuning
        weight_decay=0.01
    )
    # Initialize the trainer with appropriate parameters
    # Ensure the object has the correct model, train and validation data sets,
    # and evaluation function for computing accuracy or any other metric
    trainer = Trainer(model=model, args=training_args, train_dataset=data,
                      eval_dataset=val, compute_metrics=eval_func)
    # Run the training and evaluate on the validation set
    trainer.train()
    trainer.evaluate()
