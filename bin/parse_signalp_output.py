#!/usr/bin/env python3
import re
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

if __name__ == '__main__':

    y_true = []
    y_pred = []

    for opt in load_options:

        with open('../data/SPDS17/signalp_output/%s' % opt['file'], 'r') as f:
            content = f.read()

        mat = re.findall('SP=\'(.+?)\'', content)

        y_pred.extend([1 if el == 'YES' else 0 for el in mat])
        y_true.extend([opt['label']] * len(mat))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    print(y_true.shape)

    print('MCC:', matthews_corrcoef(y_true, y_pred))
    print('precision:', precision_score(y_true, y_pred))
    print('recall:', recall_score(y_true, y_pred))
    print('f1:', f1_score(y_true, y_pred))

    with open('../data/SPDS17/signalp_output/TMEuk.nr.out', 'r') as f:
        content = f.read()

    mat = re.findall('SP=\'(.+?)\'', content)
    mat = [1 if el == 'YES' else 0 for el in mat]
    print('fpr:', sum(mat) / len(mat))
