#!/usr/bin/env python3
import sys

import numpy as np
from sklearn.metrics import matthews_corrcoef, recall_score, precision_score, f1_score

map_db = {
    'euk': 'Euk',
    'gram-': 'Gram-',
    'gram+': 'Gram+',
}

db = sys.argv[1]

load_options = [
    { 'file': 'SP%s.nr.out' % map_db[db], 'label': 1 },
    { 'file': 'TM%s.nr.out' % map_db[db], 'label': 0 },
    { 'file': 'NC%s.nr.out' % map_db[db], 'label': 0 },
]

label = {
    'SignalPeptide': 1,
    'Transmembrane': 0,
    'Other': 0,
}

if __name__ == '__main__':

    y_true = []
    y_pred = []

    for opt in load_options:

        with open('../data/SPDS17/deepsig_output/%s' % opt['file'], 'r') as f:
            lines = f.readlines()

        pred = [line.rstrip('\n').split('\t')[1] for line in lines]
        pred = [label[tag] for tag in pred]
        true = [opt['label']] * len(pred)

        y_pred.extend(pred)
        y_true.extend(true)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    print('MCC:', matthews_corrcoef(y_true, y_pred))
    print('precision:', precision_score(y_true, y_pred))
    print('recall:', recall_score(y_true, y_pred))
    print('f1:', f1_score(y_true, y_pred))


    with open('../data/SPDS17/deepsig_output/TMEuk.nr.out', 'r') as f:
        lines = f.readlines()

    pred = [line.rstrip('\n').split('\t')[1] for line in lines]
    pred = [label[tag] for tag in pred]

    print('fpr:', sum(pred) / len(pred))
