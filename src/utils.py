from collections import OrderedDict


MAXINT32 = 2**31 - 1


def get_tf_seed(_rnd):
    return _rnd.random_integers(0, MAXINT32)


def weight_symbols_from_tfgraph(graph, n_layers=None):
    '''
    Extract weights from a tensorflow computational graph
    '''
    # Extract weight tensors from graph and evaluate them using session
    weight_symbols = OrderedDict()
    current_layer = 0
    while True:
        if n_layers is not None and n_layers == current_layer:
            break
        try:
            weight_symbols['layer_{}'.format(current_layer)] = \
                graph.get_tensor_by_name('layer_{}/weights:0'.format(current_layer))
        except KeyError as k:
            # If we dont even find one layer, something probably went wrong
            if current_layer == 0:
                raise k
            break

        current_layer += 1

    if n_layers is not None and len(weight_symbols) != n_layers:
        raise Exception('Only {} layers were retrieved from graph instead of intended n_layer={}'
                        .format(len(weight_symbols), n_layers))

    return weight_symbols


def evaluate_weights(sess, weights):
    # Evaluate all weights at once --> less calls to tf
    return OrderedDict([(name, evaluated)
                        for name, evaluated
                        in zip(weights.keys(), sess.run(list(weights.values())))
                        ])


