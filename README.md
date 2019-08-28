# dom-adv-sigU
Using Domain-Adversarial NN to Predict Cleavage Sites of Signal Peptides

## Setup environment

```
$ source config_env
```

## Install

```
$ pip3 install -r requires.txt
```

## Execute
Generate 20 models and evaluate them.

All available dataset: euk, gram-, gram+, all, bacteria

Ratio: the ratio of good and bad features. 

e.g. 0.5 for 1:1, 0.25 for 1:4

```
$ ./run.sh euk 0.5
```

All the result will be in `log/euk_0.5.txt`

## Build Feature
### Build initial features
Go to `bin`:
```
$ cd bin/
```

All available dataset: euk, gram-, gram+, all, bacteria
```
$ ./build_train_features.py euk
```
Initial training features saves in `data/features/` as `train.npy`
Evalate features saves in `data/features/` and `eval.npy`

All available dataset: euk, gram-, gram+, all, bacteria
```
$ ./build_eval_features.py euk
```
Testing features saves in `data/features/` as `test.npy`

### Build 'good' and 'bad' features using initial features
Ratio : the ratio of good and bad features. 

e.g. 0.5 for 1:1, 0.25 for 1:4
```
$ ./random_select.py 0.5
```  
'Good' and 'bad' features saves in `data/features/` `train_good.npy` and `train_bad.npy`.

## Run Experiment
### Train a model using 'Good' and 'bad' features
```
$ cd experiment/
$ ./domain_adversarial.py model_name/
```
The model saves in `models/modle_name/`


## Evaluate the model
`./get_pred.py model_name/`
