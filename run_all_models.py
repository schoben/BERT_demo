# -*- coding: utf-8 -*-
"""This file runs both models one after the other conveniently"""

import os
from model_scripts.amazon_mag_reviews import run_seq_cls
from model_scripts.conll003 import run_ner

# The paths to the data for both models
PATH_TO_SEQ_DATA: str = os.path.join('data', 'Magazine_Subscriptions_short.json.gz')
PATH_TO_NER_DATA: str = os.path.join('data', 'test.zip')
NER_FILE: str = 'test.txt'
# the directory for logging outputs
OUTPUT_DIR: str = 'results'
# The max number of samples we want from each dataset for the demo
# set equal to 0 to include all samples
SAMPLE_LIMIT: int = 800


def main():
    """The main method of this module"""
    run_seq_cls(path=PATH_TO_SEQ_DATA, o_dir=OUTPUT_DIR, trunc=SAMPLE_LIMIT)
    print('Amazon review classification complete!!!')
    run_ner(archive=PATH_TO_NER_DATA, file=NER_FILE, o_dir=OUTPUT_DIR, trunc=SAMPLE_LIMIT)
    print('Conll003 NER classification complete!!!')
    print('All models successfully run!')


if __name__ == "__main__":
    main()
