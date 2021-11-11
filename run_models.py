# -*- coding: utf-8 -*-
"""This file runs both models one after the other conveniently - it can also be set to run one at a time if needed"""

import os
import argparse
from model_scripts.amazon_mag_reviews import run_seq_cls
from model_scripts.conll003 import run_ner

# The paths to the data for both models
PATH_TO_SEQ_DATA: str = os.path.join('data', 'Magazine_Subscriptions_short.json.gz')
PATH_TO_NER_DATA: str = os.path.join('data', 'test.zip')
NER_FILE: str = 'test.txt'
# The max number of samples we want from each dataset for the demo
# set to None to include all samples
SAMPLE_LIMIT: int = 800


def main():
    """The main method of this module - collects the arguments in the parser
    and runs appropriate models"""
    # collect arguments to determine which model to run
    parser = argparse.ArgumentParser()
    # add arguments for each model type
    parser.add_argument('-n', dest='ner', action='store_true',
                        help='runs named entity model with conll003')

    parser.add_argument('-a', dest='amazon', action='store_true',
                        help='runs sequence analysis on amazon reviews')

    args = parser.parse_args()
    # This determines what gets shown
    if (not args.amazon and not args.ner) or (args.amazon and args.ner):
        # The default is that both models get to run
        print('Starting Amazon review sequence classification')
        run_seq_cls(path=PATH_TO_SEQ_DATA, trunc=SAMPLE_LIMIT)
        print('Amazon review classification complete!!!')
        print('Starting Conll003 NER classification')
        run_ner(archive=PATH_TO_NER_DATA, file=NER_FILE, trunc=SAMPLE_LIMIT)
        print('Conll003 NER classification complete!!!')
        print('All models successfully run!')

    elif args.amazon:
        print('Starting Amazon review Sequence Classification')
        run_seq_cls(path=PATH_TO_SEQ_DATA, trunc=SAMPLE_LIMIT)
        print('Amazon review classification complete!!!')

    elif args.ner:
        print('Starting Conll003 NER classification')
        run_ner(archive=PATH_TO_NER_DATA, file=NER_FILE, trunc=SAMPLE_LIMIT)
        print('Conll003 NER classification complete!!!')


if __name__ == "__main__":
    main()
