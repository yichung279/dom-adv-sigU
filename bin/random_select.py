#!/usr/bin/env python3
import sys
import random
import numpy as np
from sklearn.model_selection import train_test_split

Rate = float(sys.argv[1])
features_dir = '../data/features/'

def classify(dataset):
    s, t, nc = [], [], []
    
    for protein in dataset:
        if (protein["one_hot_label"] == [1, 0, 0]).all() == True:
            nc.append(protein)
        elif (protein["one_hot_label"] == [0, 1, 0]).all() == True:
            t.append(protein)
        else:
            s.append(protein)
    return s, t, nc

if __name__ == "__main__":
    
    train = np.load(f'{features_dir}/train.npy')  # train = train_woeval.npy
    
    train = [tr for tr in train if (tr['meta'] == 'Train')]

    s, t, nc = classify(train)
    train_s , test_s  = train_test_split(s , train_size=Rate)
    train_t , test_t  = train_test_split(t , train_size=Rate)
    train_nc, test_nc = train_test_split(nc, train_size=Rate)

    a = train_s + train_t + train_nc
    b = test_s + test_t + test_nc
    print("  a  :", len(a), "/  b:", len(b))
    print(" a_s :", len(train_s))
    print(" a_t :", len([f for f in a if (f["one_hot_label"] == [0, 1, 0]).all()==True]))
    print("a_n/c:", len([f for f in a if (f["one_hot_label"] == [1, 0, 0]).all()==True]))

    a = np.array(a, dtype = [ ('features', np.int32, (96,)),
                              ('label', np.int32),
                              ('meta', np.unicode_, 10),
                              ('residue_label', np.int32, (96,)),
                              ('cleavage_site', np.int32),
                              ('db', np.unicode_, 8),
                              ('one_hot_label', np.int32, (3, )) ])

    b = np.array(b, dtype = [ ('features', np.int32, (96,)),
                              ('label', np.int32),
                              ('meta', np.unicode_, 10),
                              ('residue_label', np.int32, (96,)),
                              ('cleavage_site', np.int32),
                              ('db', np.unicode_, 8),
                              ('one_hot_label', np.int32, (3, )) ])
    
    np.save(f"{features_dir}train_good.npy", a)
    np.save(f"{features_dir}train_bad.npy", b)
