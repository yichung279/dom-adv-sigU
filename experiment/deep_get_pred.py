import numpy as np
import sys
import os
import tensorflow as tf
import json

#from utils.models import Evaluator

SIGNATURE_INPUT = 'input'
SIGNATURE_OUTPUT = 'output'
SIGNATURE_OUTPUT_K = 'output_k'
SIGNATURE_METHOD_NAME = 'prediction'
SIGNATURE_KEY = 'prediction'
SIGNATURE_KEY_K = 'prediction_k'

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
            #output_tensor_name_k = signature[SIGNATURE_KEY_K].outputs[SIGNATURE_OUTPUT_K].name
            print(input_tensor_name)     # input_fn/feature_holder:0
            print(output_tensor_name)    # prediction_probability:0
            #print(output_tensor_name_k)    # prediction_probability_k:0
            
            # Get input, output, and config tensor.
            self.input_holder = self.session.graph.get_tensor_by_name("input_fn/feature_holder:0")
            self.prediction = self.session.graph.get_tensor_by_name(output_tensor_name)
            #self.prediction_k = self.session.graph.get_tensor_by_name(output_tensor_name_k)
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


if __name__ == "__main__":
    eval_data = np.load('../../data/features/eval.npy')
    new_evaluator = Evaluator('./test_deepsig')

    #pred, ts_vals = evaluator.eval(eval_data['features'], addition_tensors=['test/conv19/activate:0'])
    new_pred, ts_vals = new_evaluator.eval(eval_data['features'], addition_tensors=['conv5_p/activate:0'])

    np.save('./test_deepsig/conv5.npy', (ts_vals['conv5_p/activate:0']))
#    np.save('./test/new_pred.npy', new_pred)

    print(ts_vals['conv5_p/activate:0'].shape)
    print('.npy saved')

    #print([round(sum(j), 0) for i in ts_vals['test/conv19/activate:0'] for j in i])
    #print(ts_vals['test/reshape:0'].shape)
