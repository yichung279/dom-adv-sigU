#!/usr/bin/env bash

source config_env

cd models/
if [ -d $1\_$2\_model ]; then
	echo $1\_$2\_model/ 'existed'
	exit 1
fi

# preprocess
cd ../bin/
./build_train_features.py $1
./build_eval_features.py $1
./random_select.py $2

# train and eval models
cd ../experiment
counter=1
while [ "$counter" -le "20" ]
do
    ./domain_adversarial.py $1\_$2\_model/$counter/&& ./get_pred.py $1\_$2\_model/$counter/ >> ../log/$1\_$2.txt
    counter=$(($counter + 1))
done
