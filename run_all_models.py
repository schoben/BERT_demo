# -*- coding: utf-8 -*-
"""This file runs both models one after the other conveniently"""

import os
from training_scripts.amazon_mag_reviews import run_seq_cls
from training_scripts.conll003 import run_ner

# The paths to the data for both models
PATH_TO_SEQ_DATA: str = os.path.join('data', 'Magazine_Subscriptions_short.json.gz')
PATH_TO_NER_DATA: str = os.path.join('data', 'test.zip')
NER_FILE: str = 'test.txt'
OUTPUT_DIR: str = 'results'


if __name__ == "__main__":
    run_seq_cls(path=PATH_TO_SEQ_DATA, o_dir=OUTPUT_DIR)
    print('Amazon review classification complete!!!')
    run_ner(archive=PATH_TO_NER_DATA, file=NER_FILE, o_dir=OUTPUT_DIR)
    print('Conll003 NER classification complete!!!')
