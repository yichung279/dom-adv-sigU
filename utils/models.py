#!/usr/bin/env python3
import json
import os
import shutil
from math import inf
from itertools import product
from glob import glob

import numpy as np
import tensorflow as tf

from utils.dataset import k_fold_balance_split

SIGNATURE_INPUT = 'input'
SIGNATURE_OUTPUT = 'output'
SIGNATURE_OUTPUT_K = 'output_k'
SIGNATURE_METHOD_NAME = 'prediction'
SIGNATURE_KEY = 'prediction'
SIGNATURE_KEY_K = 'prediction_k'

class Learner():
    """This class is used for training a model."""

    def __init__(self, name, input_fn, model_fn, path, gpu_limit=0.4):
        """Initialize tensorflow graph.

        Initialize tensorflow graph and some options. See the following.

        Arguments:
            name (str): Variable scope for root.
            input_fn (function): Function which provide input tensor.
                                 It need to return the following four tensor:

                1. features (tf.Tensor or tf.Operation): Input tensor of model.
                2. labels   (tf.Tensor or tf.Operation): Labels to compute loss.
                3. feature_holder (tf.placeholder): placeholder to feed features.
                4. label_holder   (tf.placeholder): placeholder to feed labels.

            model_fn (function): Function which provide output tensor of model and model configuration.
                                 It need to return the following two object:

                1. output_tensor (tf.Tensor or tf.Operation): The last tensor in the model.
                2. model_config (dict): Configuration of training and testing.
                                        Keys of dict are `tf.placeholder`.
                                        Values are two-element tuple, first for traing, and second for testing.

            path (str): Save directory.
            gpu_limit (float): Maximum usage of gpu memory.
        """

        self.name = name
        self.path = path
        self.graph = tf.Graph()

        self.min_loss = None

        self.saver = None
        self.builder = None

        # Build model in `self.graph`.
        with self.graph.as_default():
            self.features, self.labels, self.feature_holder, self.label_holder=input_fn()
            
            with tf.variable_scope(name):
                self.output, model_config = model_fn(self.features)
        #        print(model_config)
        #        print('------------------------')

        ###
        #with tf.gfile.GFile("../kev_exp/freeze/pretrain.pb", "rb") as f:
        #    graph_def = tf.GraphDef()
        #    graph_def.ParseFromString(f.read())
        #    
        #with tf.Graph().as_default() as graph:
        #    tf.import_graph_def(graph_def, name='')
        #    self.features, self.labels, self.feature_holder, self.label_holder=input_fn()
        #    with tf.Session(graph=graph) as sess:
        #        print([tensor.name for tensor in grah.as_graph_def().node()])
        #    self.output, model_config = model_fn()
        ###
        
        
        
        # Setup maximum usage of gpu memory.
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_limit)
        self.session = tf.Session(graph=self.graph, config=tf.ConfigProto(gpu_options=gpu_options))

        # Save `model_config` as a json file.
        self.train_config = {}
        self.eval_config = {}

        for tensor in model_config:
            self.train_config[tensor] = model_config[tensor][0]
            self.eval_config[tensor] = model_config[tensor][1]


        if not os.path.isdir(path):
            os.makedirs(path)

        with open(self.path + '/config.json', 'w') as j:
            json.dump({
                'train': {k.name: self.train_config[k] for k in self.train_config},
                'eval': {k.name: self.eval_config[k] for k in self.eval_config}
            }, j, indent=4)

    def reset_instance_without_graph(self, path):
        """Reset all variables in a graph.

        It will reset all variables in the graph. But it will not reset graph.

        Arguments:
            path (str): New save directory.
        """

        # Reset class member.
        self.path = path
        self.min_loss = None

        self.saver = None
        self.builder = None

        # Initialize `tf.Variable`.
        with self.graph.as_default():
            self.session.run(tf.global_variables_initializer())

        if not os.path.isdir(path):
            os.makedirs(path)

        with open(self.path + '/config.json', 'w') as j:
            json.dump({
                'train': {k.name: self.train_config[k] for k in self.train_config},
                'eval': {k.name: self.eval_config[k] for k in self.eval_config}
            }, j, indent=4)

    def compile(self, optimizer, lr):
        """Initialize all variables and generate loss and train step operation.

        Arguments:
            optimizer (tf.train.Optimizer): Tensorflow optimizer.
            lr (float or tensor): Learning rate.

        Example:
            model.compile(optimizer=tf.train.AdamOptimizer, lr=1e-3)
        """

        with self.graph.as_default():

            # `tf.nn.softmax_cross_entropy_with_logits` is deprcated.
            # https://www.tensorflow.org/api_docs/python/tf/nn/softmax_cross_entropy_with_logits
            self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.labels, logits=self.output, name='cross_entropy')

            # `tf.control_dependencies` is necessary if `tf.layers.batch_normalization` is in the model
            # https://www.tensorflow.org/api_docs/python/tf/layers/batch_normalization
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                self.train_step = optimizer(lr).minimize(self.loss, name='minimize')

            # Initialize all `tf.Variable`.
            self.session.run(tf.global_variables_initializer())

    def fit(self, x_tr, y_tr, x_va, y_va, epochs, batch_size, early_stop=100, save_checkpoint=True):
        """Fit the model.

        Arguments:
            x_tr, y_tr, x_va, y_va (numpy array): Traing data and validation data.
            epochs (int): Maximum epochs for training.
            batch_size (int): Batch size for training.
            early_stop (int): Trainining will stop if validation loss in continious `early_stop` epochs didn't descent.
                              Default is `100`.
            save_checkpoint (bool): Save variables at the end of every epochs or not. Default is `True`.
        """

        self.min_loss = self.min_loss or inf
        min_loss_epoch = -1

        for epoch in range(1, epochs + 1):

            # Get the shuffle permetation of training data.
            perm = np.random.permutation(x_tr.shape[0])

            for step in range(x_tr.shape[0] // batch_size + 1):

                # Get each mini-batch training data.
                x_batch = x_tr[perm[step * batch_size:(step + 1) * batch_size]]
                y_batch = y_tr[perm[step * batch_size:(step + 1) * batch_size]]

                # train a step.
                self.session.run(self.train_step, feed_dict={
                    self.feature_holder: x_batch,
                    self.label_holder: y_batch,
                    **self.train_config,
                })


            # Get validation loss.
            val_loss = self.session.run(self.loss, feed_dict={
                self.feature_holder: x_va,
                self.label_holder: y_va,
                **self.eval_config,
            })
            val_loss = np.mean(val_loss)

            print('%d: %.6f, min loss: %.6f at %d' % (epoch, val_loss, self.min_loss, min_loss_epoch))

            # Save variables when validation loss is lower than minimum loss.
            if val_loss < self.min_loss:
                if save_checkpoint:
                    print('from %.6f to %.6f - save checkpoint to %s' % (self.min_loss, val_loss, self.path + '/checkpoint'))
                    self.save_checkpoint()

                self.min_loss = val_loss
                min_loss_epoch = epoch

            # Stop the function when validation loss of continious `early_stop` didn't descent.
            elif epoch - min_loss_epoch > early_stop:
                return

    def infer(self, data):
        """Evaluate model output.

        Arguments:
            data (numpy array): A batch of input data.

        Returns:
            res (numpy array): Model output of input data.
        """

        with self.graph.as_default():
            res = self.session.run(tf.nn.softmax(self.output), feed_dict = {self.feature_holder: data, **self.eval_config})

        return res

    def save_checkpoint(self):
        """Save model variables.

        Save all variables in the graph without model architecture.
        See https://www.tensorflow.org/guide/saved_model#save_variables .
        """

        if not os.path.isdir(self.path + '/checkpoint/'):
            os.makedirs(self.path + '/checkpoint/')

        if self.saver == None:
            with self.graph.as_default():
                self.saver = tf.train.Saver(tf.global_variables())
        
        self.saver.save(self.session, self.path + '/checkpoint/model.ckpt')

    def load_checkpoint(self):
        """Load model variables.

            Load all variables to the graph. Before call this function.
        You should make sure model architecture is built.
        See https://www.tensorflow.org/guide/saved_model#restore_variables .
        """

        if self.saver == None:
            with self.graph.as_default():
                self.saver = tf.train.Saver()

        self.saver.restore(self.session, self.path + '/checkpoint/model.ckpt')

    def serve(self):
        """Save model variables and architecture.

        Export the graph as tensorflow saved model. It will save both variables and model architecture.
        See https://www.tensorflow.org/guide/saved_model#build_and_load_a_savedmodel .
        """

        with self.graph.as_default():

            if self.builder == None:
                self.builder = tf.saved_model.builder.SavedModelBuilder(self.path + '/build/')

            # Generate softmax output.
            prediction = tf.nn.softmax(self.output, name='predict_probability')

            # Build `SignatureDef`.
            # See https://www.tensorflow.org/serving/signature_defs .
            inputs = {k.name: tf.saved_model.utils.build_tensor_info(k) for k in self.eval_config}
            inputs[SIGNATURE_INPUT] = tf.saved_model.utils.build_tensor_info(self.feature_holder)

            outputs = {SIGNATURE_OUTPUT: tf.saved_model.utils.build_tensor_info(prediction)}

            signature = tf.saved_model.signature_def_utils.build_signature_def(inputs, outputs, SIGNATURE_METHOD_NAME)
            self.builder.add_meta_graph_and_variables(self.session, tags=[tf.saved_model.tag_constants.SERVING], signature_def_map={SIGNATURE_KEY: signature})
            self.builder.save()

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
            print(input_tensor_name)
            print(output_tensor_name)
            #print(output_tensor_name_k)
            print("------------------------------------")

            # Get input, output, and config tensor.
            self.input_holder = self.session.graph.get_tensor_by_name(input_tensor_name)
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
        #prediction = self.session.run(self.prediction, feed_dict={self.input_holder: x_eval, **self.feed_tensors})
        addition = None

        if len(addition_tensors) > 0:
            addition_tensors = {name: self.graph.get_tensor_by_name(name) for name in addition_tensors}
            addition = self.session.run(addition_tensors, feed_dict={self.input_holder: x_eval, **self.feed_tensors})

        return prediction, addition

class GridSearchCV():

    def __init__(self, path, learner_fn, search_space):

        self.search_space = search_space
        self.learner_fn = learner_fn
        self.path = path
        self.folds = None

    def __search_space_iterator(self):

        args = []
        for k in self.search_space:
            pairs = []

            for v in self.search_space[k]:
                pairs.append((k, v))

            args.append(pairs)

        pre_iter = product(*args)

        for param_pairs in pre_iter:

            params = {}

            for p in param_pairs:
                params[p[0]] = p[1]

            yield params

    def fit(self, data, epochs, batch_size, early_stop, folds=None, x_tag='features', y_tag='residue_label'):

        if type(data) != list:
            if folds == None:
                print('Argument "folds" should not be "None" if type of argument "data" is not list')
                exit()
            data = k_fold_balance_split(data, folds=folds)

        self.folds = len(data)

        record = {
            'mean_loss': inf,
            'loss': [],
            'model_params': {},
        }

        inference = []
        label = []

        for params in self.__search_space_iterator():

            learner = self.learner_fn('%s/none' % self.path, params)
            shutil.rmtree('%s/none' % self.path)
            cv_loss = []
            cv_inference = []
            cv_label = []

            for i in range(self.folds):
                tr = np.concatenate([data[j] for j in range(self.folds) if j != i])
                va = data[i]

                learner.reset_instance_without_graph('%s/tmp/%d' % (self.path, i))
                learner.fit(
                    tr[x_tag], tr[y_tag], va[x_tag], va[y_tag],
                    batch_size = batch_size,
                    epochs = epochs,
                    early_stop = early_stop,
                )
                learner.load_checkpoint()
                learner.serve()

                cv_inference.append(learner.infer(va[x_tag]))
                cv_label.append(va['label'])

                cv_loss.append(float(learner.min_loss))

            if sum(cv_loss) / len(cv_loss) < record['mean_loss']:

                if os.path.isdir('%s/keep' % self.path):
                    shutil.rmtree('%s/keep/' % self.path)

                record['mean_loss'] = sum(cv_loss) / len(cv_loss)
                record['loss'] = list(cv_loss)
                record['model_params'] = dict(params)

                inference = cv_inference
                label = cv_label

                os.rename('%s/tmp' % self.path, '%s/keep' % self.path)
            else:
                shutil.rmtree('%s/tmp/' % self.path)

        with open('%s/best.json' % self.path, 'w') as j:
            json.dump(record, j, indent = 4)

        for f in glob('%s/keep/*' % self.path):
            shutil.move(f, self.path)
        shutil.rmtree('%s/keep' % self.path)

        return np.concatenate(inference, axis=0), np.concatenate(label, axis=0)
