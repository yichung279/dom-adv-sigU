#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import json
import math
import os
import sys

from itertools import zip_longest

from sklearn.metrics import matthews_corrcoef as MCC
from sklearn.metrics import classification_report

from utils.dataset import k_fold_balance_split, balance_split
from utils.models import Learner, Evaluator
from utils.layers import conv1d, deconv1d, avg_pool, dense

from tensorflow.python.framework import graph_util

gooddata = 'train_good.npy'
poordata = 'train_bad.npy'

sequence_length = 96
kernel_size = 9
n = 4
base = 24
lam = 2.0

SIGNATURE_INPUT = 'input'
SIGNATURE_OUTPUT = 'output'
SIGNATURE_OUTPUT_ADV = 'output_adv'
SIGNATURE_METHOD_NAME = 'prediction'
SIGNATURE_METHOD_NAME_ADV = 'prediction_adv'
SIGNATURE_KEY = 'prediction'
SIGNATURE_KEY_ADV = 'prediction_adv'

def get_input_fn(name):

    def input_fn():

        with tf.name_scope(name):
            feature_holder = tf.placeholder(tf.int32, [None, sequence_length], name='feature_holder')
            label_holder = tf.placeholder(tf.int32, [None, sequence_length], name='label_holder')
            label_holder_adv = tf.placeholder(tf.int32, [None], name='label_holder_adv')

            feature = tf.one_hot(feature_holder, depth=20, name='feature')
            label = tf.one_hot(label_holder, depth=3, name='label')
            label_adv = tf.one_hot(label_holder_adv, depth=3, name='label_adv')

        return feature, label, label_adv, feature_holder, label_holder, label_holder_adv

    return input_fn


class New_Learner():
    """ This class is used for training a model. """

    def __init__(self, name, input_fn, path, gpu_limit=0.4):
        """ Initialize tensorflow graph. """
        
        self.name = name
        self.path = path
        self.graph = tf.Graph()

        self.min_loss = None

        self.saver = None
        self.builder = None
        self.builder_adv = None
        
        with self.graph.as_default() as graph:
            self.features, self.labels, self.labels_adv, self.feature_holder, self.label_holder, self.label_holder_adv = input_fn()
            
            self.input_feature_holder = graph.get_tensor_by_name("input_fn/feature_holder:0")   # shape == (?, 96)
            self.input_feature_holder_adv = graph.get_tensor_by_name("input_fn/feature_holder:0")   # shape == (?, 96)
            self.input_feature = graph.get_tensor_by_name("input_fn/feature:0")                 # shape == (?, 96, 20)
            self.input_label_holder = graph.get_tensor_by_name("input_fn/label_holder:0") 
            self.input_label_holder_adv = graph.get_tensor_by_name("input_fn/label_holder_adv:0")
            self.input_label = graph.get_tensor_by_name("input_fn/label:0") 
            self.input_label_adv = graph.get_tensor_by_name("input_fn/label_adv:0") 
            
            self.conv1 = conv1d(self.input_feature, name="conv1_fe", filters=base, kernel_size=kernel_size)
            self.pass1 = conv1d(self.conv1, name="conv2_fe", filters=base, kernel_size=kernel_size)
            self.pool1 = avg_pool(self.pass1, name='pool1_fe', pool_size=2)

            self.conv3 = conv1d(self.pool1, name='conv3_fe', filters=base + n, kernel_size=kernel_size)
            self.pass2 = conv1d(self.conv3, name='conv4_fe', filters=base + n, kernel_size=kernel_size)
            self.out = avg_pool(self.pass2, name='pool2_fe', pool_size=2)

            ###
            self.conv5 = conv1d(self.out, name='conv5_fe', filters=base + 2 * n, kernel_size=kernel_size)
            self.pass3 = conv1d(self.conv5, name='conv6_fe', filters=base + 2 * n, kernel_size=kernel_size)
            self.pool3 = avg_pool(self.pass3, name='pool3_fe', pool_size=2)

            self.conv11 = conv1d(self.pool3, name='conv11_fe', filters=base + 3 * n, kernel_size=kernel_size)
            self.conv12 = conv1d(self.conv11, name='conv12_fe', filters=base + 3 * n, kernel_size=kernel_size)
            self.deconv2 = deconv1d(self.conv12, name='deconv2_fe', filters=base + 2 * n, kernel_size=kernel_size, stride=2)

            self.out = tf.concat([self.deconv2, self.pass3], axis=2)
            ###

            self.conv13 = conv1d(self.out, name='conv13_fe', filters=base + 2 * n, kernel_size=kernel_size)
            self.conv14 = conv1d(self.conv13, name='conv14_fe', filters=base + 2 * n, kernel_size=kernel_size)
            self.deconv3 = deconv1d(self.conv14, name='deconv3_fe', filters=base + n, kernel_size=kernel_size, stride=2)

            self.out = tf.concat([self.deconv3, self.pass2], axis=2)

            self.conv15 = conv1d(self.out, name='conv15_fe', filters=base + n, kernel_size=kernel_size)
            self.conv16 = conv1d(self.conv15, name='conv16_fe', filters=base + n, kernel_size=kernel_size)
            self.deconv4 = deconv1d(self.conv16, name='deconv4_fe', filters=base, kernel_size=kernel_size, stride=2)

            self.out = tf.concat([self.deconv4, self.pass1], axis=2)

            self.conv17 = conv1d(self.out, name='conv17_fe', filters=base, kernel_size=kernel_size)
            
            self.conv18 = conv1d(self.conv17, name="conv18_lp", filters=base, kernel_size=5) 
            self.conv19 = conv1d(self.conv18, name="conv19_lp", filters=3, kernel_size=1, act=tf.identity)
            self.output = tf.reshape(self.conv19, [-1, 3], name="output",)
             
            self.conv18_dc = conv1d(self.conv17, name="conv18_dc", filters=base, kernel_size=5)
            self.conv19_dc = conv1d(self.conv18_dc, name="conv19_dc", filters=3, kernel_size=1)
            self.out = tf.reshape(self.conv19_dc, (-1, 96 * 3), name="reshape_dc")
            self.output_adv = dense(self.out, name="logits_dc", units=3, act=tf.identity)
             
       

        # Setup maximum usage of gpu memory.
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_limit)
        self.session = tf.Session(graph=self.graph, config=tf.ConfigProto(gpu_options=gpu_options))

        # Save `model_config` as a json file.
        self.train_config = {}
        self.eval_config = {}

        if not os.path.isdir(path):
            os.makedirs(path)

        with open(self.path + '/config.json', 'w') as j:
            json.dump({
                'train': {},
                'eval': {}
            }, j, indent=4)

    def reset_instance_without_graph(self, path):
        """ Reset all variables in a graph.  

        It will reset all variables in the graph. But it will not reset graph.
        """

        # Reset class member.
        self.path = path
        self.min_loss = None

        self.saver = None
        self.builder = None
        self.builder_k = None

        # Initialize `tf.Variable`.
        with self.graph.as_default():
            self.session.run(tf.global_variables_initializer())

        if not os.path.isdir(path):
            os.makedirs(path)

        with open(self.path + '/config.json', 'w') as j:
            json.dump({
                'train': {},
                'eval': {}
            }, j, indent=4)

    def compile(self, optimizer, lr):
        """ Initialize all variables and generate loss and train step operation. """
        
        #clip_morm = 0.1
        self.loss_f = None
        with self.graph.as_default():
            
            tvars = tf.trainable_variables()
            ft_vars = [v for v in tvars if "_fe" in v.name]            
            lab_vars = [v for v in tvars if "_dc" not in v.name]
            dom_vars = [v for v in tvars if "_lp" not in v.name]

            print()
            print(" ft  updates:", ft_vars)
            print("96x3 updates:", lab_vars)
            print(" 1x3 updates:", dom_vars)
            print()

            # `tf.nn.softmax_cross_entropy_with_logits` is deprcated.
            # https://www.tensorflow.org/api_docs/python/tf/nn/softmax_cross_entropy_with_logits
            self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.labels, logits=self.output, name='cross_entropy')
            self.loss_adv = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.labels_adv, logits=self.output_adv, name='cross_entropy_adv')
            
            #grads_and_vars = optimizer.compute_gradients(loss, var_list=tf_vars)
            #clipped_grads_and_vars = [(tf.clip_by_norm(grad, clip_norm=clip_norm), var) for grad, var in grads_and_vars]
            
            self.loss_fe = - lam * self.loss_adv
            # `tf.control_dependencies` is necessary if `tf.layers.batch_normalization` is in the model
            # https://www.tensorflow.org/api_docs/python/tf/layers/batch_normalization
            self.train_step_adv = optimizer(lr).minimize(self.loss_fe, name='minimize_fe', var_list=ft_vars)
            self.train_step = optimizer(lr).minimize(self.loss, name='minimize', var_list=lab_vars)
            self.train_step_adv = optimizer(lr).minimize(self.loss_adv, name='minimize_adv', var_list=dom_vars)

            # Initialize all `tf.Variable`.
            self.session.run(tf.global_variables_initializer())


    def fit(self, x_tr, y_tr, x_va, y_va, x_tr_adv, y_tr_adv, x_va_adv, y_va_adv, epochs, batch_size, early_stop=100, save_checkpoint=True):
        """Fit the model.

        Arguments:
            x_tr, y_tr, x_va, y_va (numpy array): Traing data and validation data.
            epochs (int): Maximum epochs for training.
            batch_size (int): Batch size for training.
            early_stop (int): Trainining will stop if validation loss in continious `early_stop` epochs didn't descent.
                              Default is `100`.
            save_checkpoint (bool): Save variables at the end of every epochs or not. Default is `True`.
        """
        
        self.min_loss_good = self.min_loss or math.inf
        self.min_loss_poor = self.min_loss or math.inf
        min_loss_epoch = -1
        
        for epoch in range(1, epochs + 1):

            # Get the shuffle permetation of training data.
            perm = np.random.permutation(x_tr.shape[0])
            perm_adv = np.random.permutation(x_tr_adv.shape[0])

            for step, step_adv in zip_longest(range(x_tr.shape[0] // batch_size + 1), range(x_tr_adv.shape[0] // batch_size + 1)):
                    
                # Get each mini-batch training data.
                    x_batch_adv = x_tr_adv[perm[step_adv * batch_size:(step_adv + 1) * batch_size]]
                    y_batch_adv = y_tr_adv[perm[step_adv * batch_size:(step_adv + 1) * batch_size]]

                    # train a step.
                    self.session.run(self.train_step_adv, feed_dict={
                        self.input_feature_holder_adv: x_batch_adv,
                        self.input_label_holder_adv: y_batch_adv,
                        **self.train_config,
                    })
                    
                    if None == step: continue
                    x_batch = x_tr[perm[step * batch_size:(step + 1) * batch_size]]
                    y_batch = y_tr[perm[step * batch_size:(step + 1) * batch_size]]

                    self.session.run(self.train_step, feed_dict={
                        self.input_feature_holder: x_batch,
                        self.input_label_holder: y_batch,
                        **self.train_config,
                    })


            # Get validation loss.
            val_loss = self.session.run(self.loss, feed_dict={
                self.input_feature_holder: x_va,
                self.input_label_holder: y_va,
                **self.eval_config,
            })
            val_loss = np.mean(val_loss)
            
            val_loss_adv = self.session.run(self.loss_adv, feed_dict={
                self.input_feature_holder: x_va_adv,
                self.input_label_holder_adv: y_va_adv,
                **self.eval_config,
            })
            val_loss_adv = np.mean(val_loss_adv)

            print('Good data %d: %.6f, min loss: %.6f at %d' % (epoch, val_loss, self.min_loss_good, min_loss_epoch))
            print('Poor data %d: %.6f, min loss: %.6f at %d' % (epoch, val_loss_adv, self.min_loss_poor, min_loss_epoch))
            print('----------------------------------------------------------------')

            # Save variables when validation loss is lower than minimum loss.
            if val_loss < self.min_loss_good:
                if save_checkpoint:
                    print('from %.6f to %.6f - save checkpoint to %s' % (self.min_loss_good, val_loss, self.path + '/checkpoint'))
                    self.save_checkpoint()
                    print()

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

    def serve(self):
        """Load model variables.

            Load all variables to the graph. Before call this function.
        You should make sure model architecture is built.
        See https://www.tensorflow.org/guide/saved_model#restore_variables .
        """

        with self.graph.as_default():

            if self.builder == None:
                self.builder = tf.saved_model.builder.SavedModelBuilder(self.path + '/build/')

            # Generate softmax output.
            prediction = tf.nn.softmax(self.output, name='predict_probability')
            prediction_adv = tf.nn.softmax(self.output_adv, name='prediction_probability_adv')
            

            # Build `SignatureDef`.
            # See https://www.tensorflow.org/serving/signature_defs .
            inputs = {k.name: tf.saved_model.utils.build_tensor_info(k) for k in self.eval_config}
            inputs[SIGNATURE_INPUT] = tf.saved_model.utils.build_tensor_info(self.feature_holder)

            outputs = {SIGNATURE_OUTPUT: tf.saved_model.utils.build_tensor_info(prediction), SIGNATURE_OUTPUT_ADV: tf.saved_model.utils.build_tensor_info(prediction_adv)}

            signature = tf.saved_model.signature_def_utils.build_signature_def(inputs, outputs, SIGNATURE_METHOD_NAME)
            self.builder.add_meta_graph_and_variables(self.session, tags=[tf.saved_model.tag_constants.SERVING], signature_def_map={SIGNATURE_KEY: signature})
            self.builder.save()


if __name__ == "__main__":
    
    good_data = np.load("../data/features/" + gooddata)
    good_tr, good_va = balance_split(good_data, tr_size=0.8)
        
    poor_data = np.load("../data/features/" + poordata)
    poor_tr, poor_va = balance_split(poor_data, tr_size=0.8)
    
    lr = 1e-4
    
    ### Train a new model
    new_model = New_Learner(
            input_fn=get_input_fn(name="input_fn"),
            #model_fn=get_model_fn(n=16),
            path=f'../models/{sys.argv[1]}',
            gpu_limit=0.8,
            name="test",
            )

    new_model.compile(optimizer=tf.train.AdamOptimizer, lr=1e-4)
   
    new_model.fit(
            good_tr["features"], good_tr["residue_label"], good_va["features"], good_va["residue_label"],
            poor_tr["features"], poor_tr["label"], poor_va["features"], poor_va["label"],
            epochs=50,
            batch_size=128,
            early_stop=20,
            save_checkpoint=False, 
            )
    
    new_model.serve()
