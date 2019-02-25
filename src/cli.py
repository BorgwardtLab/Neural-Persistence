import argparse
import tensorflow as tf


def check_parameters(optimizer, optimizer_params):
    """Checks if the provided parameters match the chosen optimizer"""
    if (optimizer == 'gradient' or optimizer == 'SGD') and len(optimizer_params) != 1:
        raise ValueError('Gradient descent optimizer requires the learning rate parameters')
    elif optimizer == 'nesterov' and len(optimizer_params) != 2:
        raise ValueError('Nesterov optimizer requires the learning rate and the momentum parameters')
    elif optimizer == 'adam' and len(optimizer_params) != 4:
        raise ValueError('Adam optimizer requires the learning rate, beta1, beta2 and epsilon parameters')


def set_default_params(optimizer, worst_case=False):
    if optimizer == 'gradient' or optimizer == 'SGD':
        optimizer_params = [1e-5] if worst_case else [0.5]
    elif optimizer == 'nesterov':
        optimizer_params = [1e-05, 0.1] if worst_case else [0.01, 0.1]
    elif optimizer == 'adam':
        optimizer_params = [1e-05, 0.5, 0.9, 0.1] if worst_case else [0.0003, 0.9, 0.999, 1e-08]
    return optimizer_params


def construct_optimizer(optimizer_name, optimizer_params):
    if optimizer_name == 'gradient' or optimizer_name == 'SGD':
        optimizer = tf.train.GradientDescentOptimizer(optimizer_params[0])
    elif optimizer_name == 'nesterov':
        optimizer = tf.train.MomentumOptimizer(learning_rate=optimizer_params[0], momentum=optimizer_params[1],use_nesterov=True)
    elif optimizer_name == 'adam':
        optimizer = tf.train.AdamOptimizer(learning_rate=optimizer_params[0], beta1=optimizer_params[1], beta2=optimizer_params[2], epsilon=optimizer_params[3])
    else:
        raise ValueError('{} is not a correct optimizer'.format(optimizer_name))
    return optimizer


def parse_initialization(initialization_name):
    if initialization_name == 'random':
        chosen_initializer = tf.random_normal_initializer(stddev=1e-5)
    elif initialization_name == 'xavier':
        chosen_initializer = tf.contrib.layers.xavier_initializer()
    else:
        chosen_initializer = tf.zeros_initializer
    return chosen_initializer
