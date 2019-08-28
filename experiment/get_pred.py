#!/usr/bin/env python3

import numpy as np
import sys
import os
import tensorflow as tf
import json
import math

SIGNATURE_INPUT = 'input'
SIGNATURE_OUTPUT = 'output'
SIGNATURE_OUTPUT_ADV = 'output_adv'
SIGNATURE_METHOD_NAME = 'prediction'
SIGNATURE_KEY = 'prediction'
SIGNATURE_KEY_ADV = 'prediction_adv'

class Evaluator():
    """This class is uesd for evaluate tensors in a model."""

    def __init__(self, path):
        """Load savedmodel into a graph.

        Arguments:
            path (str): A generated directory by a `Learner` instance.
        """

        # Load configuration for testing.
        self.graph = tf.Graph()

        with open('%s/config.json' % path, 'r') as j:
            tensor_config = json.load(j)['eval']

        with self.graph.as_default():
            self.session = tf.Session()

            # Load savedmodel.
            meta_graph_def = tf.saved_model.loader.load(self.session, [tf.saved_model.tag_constants.SERVING], '%s/build/' % path)
            signature = meta_graph_def.signature_def

            input_tensor_name = signature[SIGNATURE_KEY].inputs[SIGNATURE_INPUT].name
            output_tensor_name = signature[SIGNATURE_KEY].outputs[SIGNATURE_OUTPUT].name
            #print(input_tensor_name)     # input_fn/feature_holder:0
            #print(output_tensor_name)    # prediction_probability:0
            
            # Get input, output, and config tensor.
            self.input_holder = self.session.graph.get_tensor_by_name("input_fn/feature_holder:0")
            self.prediction = self.session.graph.get_tensor_by_name(output_tensor_name)
            self.feed_tensors = {self.session.graph.get_tensor_by_name(k): tensor_config[k] for k in tensor_config}

    def eval(self, x_eval, addition_tensors=[]):
        """Evaluate value of tensor with input data.

        It will evaluate output tensor of the model.
        If addition_tensors is not empty,
        it will also evaluate all tensors which names are in `addition_tensors`.

        Arguments:
            x_eval (numpy array): Input data.
            addition_tensors (list): List of tensor name you want to evaluate.

        Returns:
            prediction (numpy array): Output value of the model with input data.
            addition (dict): Dict to pack values of addition tensors.
                             Its key is the name of tensor and value is the evaluate value of tensor.
                             It is `None` if `addition_tensors` is empty.

        Examples:
            x_eval = np.ones((1000, 20))
            evaluator.eval(x_eval, addition_tensors=['model/dense1/activate:0', 'model/dense2/activate:0'])
        """
        
        prediction = self.session.run(self.prediction, feed_dict={self.input_holder: x_eval, **self.feed_tensors})
        addition = None

        if len(addition_tensors) > 0:
            addition_tensors = {name: self.graph.get_tensor_by_name(name) for name in addition_tensors}
            addition = self.session.run(addition_tensors, feed_dict={self.input_holder: x_eval, **self.feed_tensors})

        return prediction, addition


def pred_index(sequences):
    for i, j in enumerate(sequences):
        if j[0] > j[2]:   
            return i-1
    return -1

if __name__ == "__main__":
    eval_data = np.load('../data/features/eval.npy')
    evaluator = Evaluator(f'../models/{sys.argv[1]}')

    pred, ts_vals = evaluator.eval(eval_data['features'], addition_tensors=['conv19_lp/activate:0'])

    predict = ts_vals['conv19_lp/activate:0']

    count_dict = {}
    pred_cleavage_count, eval_cleavage_count = 0, 0
    for i, (each_predict, cleavage_site) in enumerate(zip(predict, eval_data['cleavage_site'])):
        pred_site = pred_index(each_predict)
        
        if -1 != pred_site:
            pred_cleavage_count += 1
        if -1 != cleavage_site:
            eval_cleavage_count += 1

        if (pred_site > 0 and cleavage_site > 0):
            if abs(pred_site - cleavage_site) not in count_dict.keys():
                count_dict[abs(pred_site - cleavage_site)] = 1
            else:
                count_dict[abs(pred_site - cleavage_site)] += 1
    
    if  sum(count_dict.values()) > 0:
        mse = sum([i*i*count_dict[i] for i in count_dict.keys()]) / sum(count_dict.values())
        rmse = math.sqrt(sum([i*i*count_dict[i] for i in count_dict.keys()]) / sum(count_dict.values()))
        mae = sum([i*count_dict[i] for i in count_dict.keys()]) / sum(count_dict.values())
        
        print('{} sequences have been predicted cleavage site by Dom-Adv sigUNet.'.format(pred_cleavage_count))
        print('{} sequences have cleavage sites in Evaluation Train.'.format(eval_cleavage_count))
        print('--------------------------------------------------------------------------')
        print('{} sequences both have cleavage sites.'.format(sum(count_dict.values())))
        print('Overall MSE: {}'.format(sum([i*i*count_dict[i] for i in count_dict.keys()]) / sum(count_dict.values())))
        print('Overall RMSE: {}'.format(math.sqrt(sum([i*i*count_dict[i] for i in count_dict.keys()]) / sum(count_dict.values()))))
        print('Overall MAE: {}'.format(sum([i*count_dict[i] for i in count_dict.keys()]) / sum(count_dict.values())))

#        with open('results/bac/result_1.txt', 'a') as f:
#            f.write(f"{mse}, {rmse}\n")
