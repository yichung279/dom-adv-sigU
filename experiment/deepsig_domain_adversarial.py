import tensorflow as tf
import numpy as np
import json
import math 
import os

from itertools import zip_longest

from sklearn.metrics import matthews_corrcoef as MCC
from sklearn.metrics import classification_report

from utils.dataset import k_fold_balance_split, balance_split
from utils.models import Learner, Evaluator
from utils.layers import conv1d, deconv1d, avg_pool, dense

from tensorflow.python.framework import graph_util

sequence_length = 96
kernel_size = 11
lam = 2.0
#ckpt_dir = "./exp_data/nested_cv_mcc_0.901_fpr_0.037/0/0/checkpoint/"
ckpt_dir = "../restore/adv_model/test/checkpoint/"

    
SIGNATURE_INPUT = 'input'
SIGNATURE_OUTPUT = 'output'
SIGNATURE_OUTPUT_K = 'output_k'
SIGNATURE_METHOD_NAME = 'prediction'
SIGNATURE_METHOD_NAME_K = 'prediction_k'
SIGNATURE_KEY = 'prediction'
SIGNATURE_KEY_K = 'prediction_k'

def get_input_fn(name):

    def input_fn():

        with tf.name_scope(name):
            feature_holder = tf.placeholder(tf.int32, [None, sequence_length], name='feature_holder')
            feature_holder_k = tf.placeholder(tf.int32, [None, sequence_length], name='feature_holder_k')
            label_holder = tf.placeholder(tf.int32, [None, sequence_length], name='label_holder')
            label_holder_k = tf.placeholder(tf.int32, [None], name='label_holder_k')

            feature = tf.one_hot(feature_holder, depth=20, name='feature')
            feature_k = tf.one_hot(feature_holder_k, depth=20, name='feature_k')
            label = tf.one_hot(label_holder, depth=3, name='label')
            label_k = tf.one_hot(label_holder_k, depth=3, name='label_k')

        return feature, label, label_k, feature_holder, label_holder, label_holder_k

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
        self.builder_k = None
        

        # Restore and Build model in `self.graph`.
        with tf.gfile.GFile('../../freeze/pretrain_awo_conv17_1.pb', 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        with self.graph.as_default() as graph:
            tf.import_graph_def(graph_def, name='')
            self.features, self.labels, self.labels_k, self.feature_holder, self.label_holder, self.label_holder_k = input_fn()
            
            self.input_feature_holder = graph.get_tensor_by_name("input_fn/feature_holder:0")   # shape == (?, 96)
            self.input_feature_holder_k = graph.get_tensor_by_name("input_fn/feature_holder:0")   # shape == (?, 96)
            self.input_feature = graph.get_tensor_by_name("input_fn/feature:0")                 # shape == (?, 96, 20)
            self.input_label_holder = graph.get_tensor_by_name("input_fn/label_holder:0") 
            self.input_label_holder_k = graph.get_tensor_by_name("input_fn/label_holder_k:0")
            self.input_label = graph.get_tensor_by_name("input_fn/label:0") 
            self.input_label_k = graph.get_tensor_by_name("input_fn/label_k:0") 
            
            #print([tensor.name for tensor in graph.as_graph_def().node])
            
             
            self.conv1 = conv1d(self.input_feature, name="conv1_f", filters=16, kernel_size=kernel_size)
            self.pool1 = avg_pool(self.conv1, name="pool1_f", pool_size=2)

            self.conv2 = conv1d(self.pool1, name="conv2_f", filters=32, kernel_size=kernel_size)
            self.pool2 = avg_pool(self.conv2, name="pool2_f", pool_size=2)

            self.conv3 = conv1d(self.pool2, name="conv3_f", filters=64, kernel_size=kernel_size)
            self.pool3 = avg_pool(self.conv3, name="pool3_f", pool_size=2)

            self.conv4 = conv1d(self.pool3, name="conv4_p", filters=64, kernel_size=kernel_size)
            
            self.add1 = deconv1d(self.conv4, kernel_size=kernel_size, filters=64, name='deconv1', stride=2)
            self.added = tf.add(self.conv3, self.add1)
            self.conv5 = deconv1d(self.added, name='conv5_p', kernel_size=kernel_size, filters=3, stride=4, act=tf.identity)
            self.output = tf.reshape(self.conv5, [-1, 3], name='output')
            
            self.conv4_k = conv1d(self.pool3, name="conv4_d", filters=64, kernel_size=kernel_size)
            self.add1_k = deconv1d(self.conv4_k, kernel_size=kernel_size, filters=64, name='deconv1_d', stride=2)
            self.added_k = tf.add(self.conv3, self.add1_k)
            self.deconv_k = deconv1d(self.added_k, name='conv5_d', kernel_size=kernel_size, filters=3, stride=4, act=tf.identity)
            self.out = tf.reshape(self.deconv_k, (-1, 96 * 3), name='reshape_d')
            self.output_k = dense(self.out, name='logits_d', units=3, act=tf.identity)

            #with tf.variable_scope(name):
            #    self.output, model_config = model_fn(self.features)
            #    print(model_config)
            #    print('------------------------')
       

        # Setup maximum usage of gpu memory.
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_limit)
        self.session = tf.Session(graph=self.graph, config=tf.ConfigProto(gpu_options=gpu_options))

        # Save `model_config` as a json file.
        self.train_config = {}
        self.eval_config = {}

        #for tensor in model_config:
        #    self.train_config[tensor] = model_config[tensor][0]
        #    self.eval_config[tensor] = model_config[tensor][1]


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
        clip_morm = 0.1
        self.loss_f = None
        with self.graph.as_default():
            
            tvars = tf.trainable_variables()
            ft_vars = [v for v in tvars if "_f" in v.name]            
            pre_vars = [v for v in tvars if "_d" not in v.name]
            dom_vars = [v for v in tvars if "_p" not in v.name]

            print()
            print(" ft  updates:", ft_vars)
            print("96x3 updates:", pre_vars)
            print(" 1x3 updates:", dom_vars)
            print()

            # `tf.nn.softmax_cross_entropy_with_logits` is deprcated.
            # https://www.tensorflow.org/api_docs/python/tf/nn/softmax_cross_entropy_with_logits
            self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.labels, logits=self.output, name='cross_entropy')
            self.loss_k = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.labels_k, logits=self.output_k, name='cross_entropy_k')
            
            #grads_and_vars = optimizer.compute_gradients(loss, var_list=tf_vars)
            #clipped_grads_and_vars = [(tf.clip_by_norm(grad, clip_norm=clip_norm), var) for grad, var in grads_and_vars]
            
            self.loss_f = - lam * self.loss_k
            # `tf.control_dependencies` is necessary if `tf.layers.batch_normalization` is in the model
            # https://www.tensorflow.org/api_docs/python/tf/layers/batch_normalization
            self.train_step_k = optimizer(lr).minimize(self.loss_f, name='minimize_f', var_list=ft_vars)
            self.train_step = optimizer(lr).minimize(self.loss, name='minimize', var_list=pre_vars)
            self.train_step_k = optimizer(lr).minimize(self.loss_k, name='minimize_k', var_list=dom_vars)

            # Initialize all `tf.Variable`.
            self.session.run(tf.global_variables_initializer())


    def fit(self, x_tr, y_tr, x_va, y_va, x_tr_k, y_tr_k, x_va_k, y_va_k, epochs, batch_size, early_stop=100, save_checkpoint=True):
        """Fit the model.

        Arguments:
            x_tr, y_tr, x_va, y_va (numpy array): Traing data and validation data.
            epochs (int): Maximum epochs for training.
            batch_size (int): Batch size for training.
            early_stop (int): Trainining will stop if validation loss in continious `early_stop` epochs didn't descent.
                              Default is `100`.
            save_checkpoint (bool): Save variables at the end of every epochs or not. Default is `True`.
        """
        
        self.min_loss_u = self.min_loss or math.inf
        self.min_loss_k = self.min_loss or math.inf
        min_loss_epoch = -1
        
        for epoch in range(1, epochs + 1):

            # Get the shuffle permetation of training data.
            perm = np.random.permutation(x_tr.shape[0])
            perm_k = np.random.permutation(x_tr_k.shape[0])

            for step, step_k in zip_longest(range(x_tr.shape[0] // batch_size + 1), range(x_tr_k.shape[0] // batch_size + 1)):

                # Get each mini-batch training data.
                x_batch_k = x_tr_k[perm_k[step_k * batch_size:(step_k + 1) * batch_size]]
                y_batch_k = y_tr_k[perm_k[step_k * batch_size:(step_k + 1) * batch_size]]

                # train a step.
                self.session.run(self.train_step_k, feed_dict={
                    self.input_feature_holder_k: x_batch_k,
                    self.input_label_holder_k: y_batch_k,
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
            
            val_loss_k = self.session.run(self.loss_k, feed_dict={
                self.input_feature_holder: x_va_k,
                self.input_label_holder_k: y_va_k,
                **self.eval_config,
            })
            val_loss_k = np.mean(val_loss_k)

            print('SignalP %d: %.6f, min loss: %.6f at %d' % (epoch, val_loss, self.min_loss_u, min_loss_epoch))
            print('SPDS17  %d: %.6f, min loss: %.6f at %d' % (epoch, val_loss_k, self.min_loss_k, min_loss_epoch))
            print('----------------------------------------------------------------')

            # Save variables when validation loss is lower than minimum loss.
            if val_loss < self.min_loss_u:
                if save_checkpoint:
                    print('from %.6f to %.6f - save checkpoint to %s' % (self.min_loss_u, val_loss, self.path + '/checkpoint'))
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
            prediction_k = tf.nn.softmax(self.output_k, name='prediction_probability_k')
            

            # Build `SignatureDef`.
            # See https://www.tensorflow.org/serving/signature_defs .
            inputs = {k.name: tf.saved_model.utils.build_tensor_info(k) for k in self.eval_config}
            inputs[SIGNATURE_INPUT] = tf.saved_model.utils.build_tensor_info(self.feature_holder)

            outputs = {SIGNATURE_OUTPUT: tf.saved_model.utils.build_tensor_info(prediction), SIGNATURE_OUTPUT_K: tf.saved_model.utils.build_tensor_info(prediction_k)}

            signature = tf.saved_model.signature_def_utils.build_signature_def(inputs, outputs, SIGNATURE_METHOD_NAME)
            self.builder.add_meta_graph_and_variables(self.session, tags=[tf.saved_model.tag_constants.SERVING], signature_def_map={SIGNATURE_KEY: signature})
            #signature_k = tf.saved_model.signature_def_utils.build_signature_def(inputs, outputs, SIGNATURE_METHOD_NAME_K)
            #self.builder.add_meta_graph_and_variables(self.session, tags=[tf.saved_model.tag_constants.SERVING], signature_def_map={SIGNATURE_KEY_K: signature_k})
            self.builder.save()


if __name__ == "__main__":
    
    data = np.load("../../data/features/a_woeval.npy")
    tr, va = balance_split(data, tr_size=0.8)
        
    data_k = np.load("../../data/features/b_woeval.npy")
    tr_k, va_k = balance_split(data_k, tr_size=0.8)
    
    lr = 1e-4
    
    ### Train a new model
    new_model = New_Learner(
            input_fn=get_input_fn(name="input_fn"),
            #model_fn=get_model_fn(n=16),
            path="./test_deepsig",
            gpu_limit=0.8,
            name="test",
            )

    new_model.compile(optimizer=tf.train.AdamOptimizer, lr=1e-4)
 
    new_model.fit(
            tr["features"], tr["residue_label"], va["features"], va["residue_label"],
            tr_k["features"], tr_k["label"], va_k["features"], va_k["label"],                
            epochs=50,
            batch_size=128,
            early_stop=20,
            save_checkpoint=False, 
            )
    
    new_model.serve()
