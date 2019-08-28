#!/usr/bin/env python3
import tensorflow as tf

def get_variable(name, shape=None, init_param=None, trainable=True):
    """Generate a variable.

    Generate a variable with truncated normal distribution or initial value.

    Arguments:
        name (str): Name of variable.
        shape (tuple or list): Shape of variable.
        init_param (numpy array): Initial value of variable. It this option is set, `shape` will be ignore.
        trainable (bool): Generated variable is trainable or not.

    Returns:
        variable (tf.Variable): A variable which match requirement.
    """
    initializer = tf.truncated_normal(shape=shape, stddev=0.1) if init_param is None else tf.constant(init_param)

    return tf.get_variable(name, initializer=initializer, trainable=trainable)

def conv1d(tensor, name, filters=None, kernel_size=None, init_weight=None, init_bias=None, trainable=True, act=tf.nn.relu):
    """1d convoluation operation.

    Arguments:
        tensor (tf.Tensor or tf.Operation): Input tensor to do convoluation.
                                            Its shape need to be `(batch_size, features, channels)`.
        name (str): Name of this operation.
        filters (int): Number of filters.
        kernel_size (int): Length of 1d kernel.
        init_weight (numpy array): Initial value of weight.
        init_bias (numpy array): Initial value of bias.
        trainable (bool): Variable in this operation is traniable or not.
        act (tensorflow activation function): Activation function after convoluation. Default is `tf.nn.relu`.

    Returns:
        operation (tf.Operation): An operation generated by convoluation.
    """

    in_channels = int(tensor.get_shape()[2])

    with tf.name_scope(name):

        with tf.variable_scope(name):
            weight = get_variable(name='weight', shape=(kernel_size, in_channels, filters), init_param=init_weight, trainable=trainable)
            bias = get_variable(name='bias', shape=(filters, ), init_param=init_bias, trainable=trainable)

        conv1d = tf.nn.conv1d(tensor, stride=1, filters=weight, padding='SAME', name='convoluation')
        bias_add = tf.add(conv1d, bias, name='bias_add')
        activate = act(bias_add, name='activate')

    return activate

def avg_pool(tensor, name, pool_size=2, strides=2):
    """1d average pooling operation.

    Arguments:
        tensor (tf.Tensor or tf.Operation): Input tensor to do convoluation.
                                            Its shape need to be `(batch_size, features, channels)`.
        name (str): Name of this operation.
        pool_size (int): Length of a pooling. Default is `2`.
        strides (int): Distance between two pooling steps. Default is `2`.

    Returns:
        operation (tf.Operation): An operation generated by average pooling.
    """
    return tf.nn.pool(tensor, window_shape=(pool_size,), padding='VALID', strides=(strides,), pooling_type='AVG', name=name)

def deconv1d(tensor, name, filters, kernel_size, stride=1, act=tf.nn.relu):
    """1d deconvoluation operation.

    Arguments:
        tensor (tf.Tensor or tf.Operation): Input tensor to do convoluation.
                                            Its shape need to be `(batch_size, features, channels)`.
        name (str): Name of this operation.
        filters (int): Number of filters.
        kernel_size (int): Length of 1d kernel.
        stride (int): Distance between two deconvoluation steps.
        act (tensorflow activation function): Activation function after convoluation. Default is `tf.nn.relu`.

    Returns:
        operation (tf.Operation): An operation generated by deconvoluation.
    """

    in_length = int(tensor.get_shape()[1])
    in_channels = int(tensor.get_shape()[2])

    with tf.name_scope(name):

        with tf.variable_scope(name):

            w = get_variable(name='weight', shape=(kernel_size, filters, in_channels))
            b = get_variable(name='bias', shape=(filters,))

        output_shape = tf.stack([tf.shape(tensor)[0], tf.constant(in_length * stride), tf.constant(filters)],
                                axis=0, name='output_shape')

        out = tf.contrib.nn.conv1d_transpose(tensor, filter=w, stride=stride, output_shape=output_shape, name='deconvolution')
        out = tf.add(out, b, name='bias_add')
        out = act(out, 'activate')

    return out

def dense(tensor, name, units=None, init_weight=None, init_bias=None, trainable=True, act=tf.nn.relu):
    """Dense operation.

    Arguments:
        tensor (tf.Tensor or tf.Operation): Input tensor to do convoluation.
        name (str): Name of this operation.
        units (int): Number of neurons. It will be ignore if `init_weight` and `init_bias` are set.
        init_weight (numpy array): Initial value of weight.
        init_bias (numpy array): Initial value of bias.
        trainable (bool): Variable in this operation is traniable or not.
        act (tensorflow activation function): Activation function after convoluation. Default is `tf.nn.relu`.

    Returns:
        operation (tf.Operation): An operation generated by dense operation.
    """

    in_dims = int(tensor.get_shape()[1])

    with tf.name_scope(name):

        with tf.variable_scope(name):
            weight = get_variable(name='weight', shape=(in_dims, units), init_param=init_weight, trainable=trainable)
            bias = get_variable(name='bias', shape=(units,), init_param=init_bias, trainable=trainable)

        dense = tf.matmul(tensor, weight, name='linear')
        bias_add = tf.add(dense, bias, name='bias_add')
        activate = act(bias_add, name='activate')

    return activate
