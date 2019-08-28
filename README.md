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

Ratio: the ratio of good dVata and bad data. e.g. 0.5 for 1:1, 0.25 for 1:4

```
$ ./run.sh euk 0.5
```

All the result will be in `log/euk_0.5.txt`

## Build Feature
### Generate .npy file contain two kinds of dataset, SignalP / SPDS17
Go to directory:`bin`:
```
$ cd bin/
```

All available dataset: euk, gram-, gram+, all, bacteria
```
$ ./build_train_features.py euk
```
Output : `train.npy` and `eval.npy` in `data/features/`

All available dataset: euk, gram-, gram+, all, bacteria
```
$ ./build_eval_features.py euk
```
Output : `test.npy` in `data/features/`

### Build good data and bad data, using output of `./build_train_features.py`
Ratio : the ratio of good data and bad data. e.g. 0.5 for 1:1, 0.25 for 1:4
```
$ ./random_select.py 0.5
```  
Output : `train_good.npy` and `train_bad.npy`.

## Run Experiment
### Train a model using `train_good.npy` and `train_bad.npy`. Save the model in `models/[modle_name]/`
```
$ cd experiment/
$ ./domain_adversarial.py model_name/
```

## Evaluate the model
`./get_pred.py model_name/`
