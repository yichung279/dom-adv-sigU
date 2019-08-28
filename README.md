# dom-adv-sigU
Using Domain-Adversarial NN to Predict Cleavage Sites of Signal Peptides

## Setup environment

`source config_env`

## Install

`pip3 install -r requires.txt`

## Execute
Generate 20 models and eval them.

`./run.sh [dataset] [ratio]`
  - [dataset]: euk, gram-, gram+, all, bacteria
  - [ratio]: the ratio of good data and bad data. e.g. 0.5 for 1:1, 0.25 for 1:4

All the result will be in `log/[dataset]_[ratio].txt`

## Build Feature
- Generate .npy file contain two kinds of dataset, SignalP / SPDS17

  - `cd bin/`

  - `./build_train_features.py [dataset]`
    - [dataset]: euk, gram-, gram+, all, bacteria
    - Output : `train.npy` and `eval.npy` in `data/features/`

  - `./build_eval_features.py [dataset]`
    - [dataset]: euk, gram-, gram+, all, bacteria
    - Output : `test.npy` in `data/features/`

- Build good data and bad data, using output of `./build_train_features.py`

 - `./random_select.py [ratio]`  
  - [rate] : the ratio of good data and bad data. e.g. 0.5 for 1:1, 0.25 for 1:4
  - Output : `train_good.npy` and `train_bad.npy`.

## Run Experiment
- Use `train_good.npy` and `train_bad.npy` to train a model. Save the model in `models/[modle_name]/`
  - `cd experiment/`
  - `./domain_adversarial.py [model_name]`

- Evaluate the model
  - `./get_pred.py [model_name]`
