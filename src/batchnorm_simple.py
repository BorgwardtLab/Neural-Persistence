import numpy as np
import tensorflow as tf
from collections import defaultdict

from sacred import Experiment
from sacred.stflow import LogFileWriter
from tensorflow.examples.tutorials.mnist import input_data
from .cli import construct_optimizer, parse_initialization
from .model_definitions import build_deep_nn_model
from .utils import evaluate_weights, weight_symbols_from_tfgraph, get_tf_seed
from .tda import PerLayerCalculation


ex = Experiment("Dropout_simple")


def get_total_persistence(weights):
    plc    = PerLayerCalculation()
    values = plc(weights)

    return values['global']['accumulated_total_persistence'],\
           values['global']['accumulated_total_persistence_normalized']


@ex.config
def cfg():
    run = {
        'batch_size': 32,
        'epochs': 40,
        'runs': 50,
        'early_stopping': False, # monitor validation loss and stop according to patience in case it decreases
        'patience': 4           # in epochs, can also be expressed in 0.25 steps
    }

    optimizer = {
        'optimizer_name': 'adam',
        'parameters': [0.0003, 0.9, 0.999, 1e-08]
    }

    model = {
        "architecture": [650, 650], # Architecture of the network
        "activation_fn": 'relu',      # Activation function (default: relu, other possibilities: tanh, sigmoid)
        'initialization'       : 'xavier',
        "batch_normalization": True           # Rate at which neurons should be randomly set to zero
    }


@ex.command(prefix='model')
def build_model(architecture, activation_fn, initialization, batch_normalization):
    initializer = parse_initialization(initialization)
    return build_deep_nn_model(architecture, activation_fn=activation_fn,
                               initializer=initializer, batch_normalization=batch_normalization)


@ex.command(prefix='optimizer')
def get_optimizer(optimizer_name, parameters):
    return construct_optimizer(optimizer_name, parameters)




@ex.automain
@LogFileWriter(ex)
def main(_rnd, run, model):
    # Build the model
    placeholders, metrics = build_model()

    # Define optimizer and training step
    optimizer = get_optimizer()
    # Need this to track moving average during training for batch norm
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = optimizer.minimize(metrics['loss'])

    # initialize variables
    inits = [tf.global_variables_initializer(), tf.local_variables_initializer()]

    results = defaultdict(list)

    for current_run in range(run['runs']):
        # Make TensorFlow random again!
        tf.set_random_seed(get_tf_seed(_rnd))

        mnist = input_data.read_data_sets('MNIST_data/', one_hot=True, seed=get_tf_seed(_rnd))

        with tf.Session() as sess:
            # initialize variables
            sess.run(inits)

            # Compute globally required quantities
            X = placeholders['x']
            Y = placeholders['y']

            iterations_per_epoch = int(mnist.train.num_examples/run['batch_size'])
            total_iterations = (run['epochs']+1)*iterations_per_epoch
            quarter_epoch = int(np.ceil(iterations_per_epoch/4))     # round up
            weights = weight_symbols_from_tfgraph(sess.graph)

            best_val_loss = float('inf')
            patience_counter = 0
            stop = False

            # store these for results
            selected_val_acc = 0
            selected_test_acc = 0
            selected_step = 0
            selected_total_persistence_normalized = 0
            for epoch in range(run['epochs']):
                if stop:
                    break

                for i in range(iterations_per_epoch):
                    batch_x, batch_y = mnist.train.next_batch(run['batch_size'])
                    _ = sess.run([train_step], feed_dict={X: batch_x, Y: batch_y})

                    if i % quarter_epoch == 0 or (i == iterations_per_epoch-1 and epoch == run['epochs']-1):
                        step = epoch*iterations_per_epoch+i

                        # compute statistics
                        acc_train = sess.run(metrics['accuracy'], feed_dict={
                            X: batch_x, Y: batch_y})

                        # check performance on validation_set
                        acc_val = sess.run(metrics['accuracy'], feed_dict={
                            X: mnist.validation.images, Y: mnist.validation.labels})
                        loss_val = sess.run(metrics['loss'], feed_dict={
                            X: mnist.validation.images, Y: mnist.validation.labels})

                        acc_test = sess.run(metrics['accuracy'], feed_dict={
                            X: mnist.test.images, Y: mnist.test.labels})

                        print("Accuracy for epoch {:.2f} of run {}/{}: {}/{}".format(epoch + i/float(iterations_per_epoch), current_run+1, run['runs'], acc_train, acc_val))

                        # see if early stopping criterion is met
                        if run['early_stopping']:
                            if loss_val < best_val_loss:
                                best_val_loss = loss_val
                                selected_val_acc = acc_val
                                selected_test_acc = acc_test
                                selected_step = step
                                patience_counter = 0
                                evaluated_weights = evaluate_weights(sess, weights)
                            else:
                                patience_counter += 0.25
                                if patience_counter > run['patience']:
                                    print('Early stopping criterion found, stopping training')
                                    stop = True
                                    break
            if stop:
                # we used early stopping to stop this run
                results['val_loss'].append(best_val_loss)
                results['val_acc'].append(selected_val_acc)
                results['test_acc'].append(selected_test_acc)
                results['step'].append(selected_step)
                _, selected_total_persistence_normalized = get_total_persistence(evaluated_weights)
                results['total_persistence_normalized'].append(selected_total_persistence_normalized)
            else:
                # early stopping was not used/triggered
                results['val_loss'].append(loss_val)
                results['val_acc'].append(acc_val)
                results['test_acc'].append(acc_test)
                results['step'].append(step)
                evaluated_weights = evaluate_weights(sess, weights)
                _, selected_total_persistence_normalized = get_total_persistence(evaluated_weights)
                results['total_persistence_normalized'].append(selected_total_persistence_normalized)

    return results
