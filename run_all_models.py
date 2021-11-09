# -*- coding: utf-8 -*-
"""This file runs both models one after the other conveniently"""


from training_scripts.amazon_mag_reviews import run_seq_cls
from training_scripts.conll003 import run_ner


if __name__ == "__main__":
    run_seq_cls()
    print('Amazon review classification complete!!!')
    run_ner()
    print('Conll003 NER classification complete!!!')
