import tensorflow as tf


def activation_layer(input_tensor, act_function, name):
    if act_function == 'relu':
        return tf.nn.relu(input_tensor, name)
    elif act_function == 'tanh':
        return tf.tanh(input_tensor, name)
    elif act_function == 'linear':
        return input_tensor
    else:
        return tf.sigmoid(input_tensor, name)


def apply_batch_normalization(input_tensor, act_function, name):
    return tf.layers.batch_normalization(input_tensor, name=name)


def apply_dropout(input_tensor, dropout_rate, training_placeholder, name):
    return tf.layers.dropout(input_tensor, dropout_rate, training=training_placeholder)


def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)


def build_layer(input_tensor, n_unit, scope, activation_fn='relu',
                initializer=tf.contrib.layers.xavier_initializer(),
                batch_normalization=False, dropout_rate=0, training_placeholder=None):
    """
    Create one fully connected layer with n_unit hidden units
    """
    if dropout_rate != 0 and training_placeholder is None:
        raise ValueError('training_placeholder must be supplied if using dropout!')

    if batch_normalization and dropout_rate != 0:
        raise ValueError('batch norm and droput should not be used together: https://arxiv.org/abs/1801.05134')

    # Get the shape of the input tensor
    num_rows = tf.shape(input_tensor)[0]
    num_cols = input_tensor.shape[1]

    # Pad with a one for bias direct integration
    h_tiled = tf.tile([[1.]], tf.stack([num_rows, 1]))
    x_b = tf.concat([input_tensor, h_tiled], axis=1)
    with tf.variable_scope(scope):
        weights = tf.get_variable(name='weights', shape=[num_cols+1, n_unit], initializer=initializer)
        variable_summaries(weights)
        multipl = tf.matmul(x_b, weights, name='multipl')
        output = activation_layer(multipl, activation_fn, name='activ')

        # only use batchnorm and dropout for non output layers
        if activation_fn != 'linear':
            if dropout_rate != 0:
                output = apply_dropout(output, dropout_rate, training_placeholder, name='dropout')

            if batch_normalization:
                # only applys if activation function is not linear
                output = apply_batch_normalization(output, activation_fn, 'batch_norm')

    return output


def build_deep_nn_model(architecture, graph=tf.get_default_graph(), x=None, y=None, **kwargs):
    """Build a deep fully connected NN model. kwargs are passed to build layer function."""
    # Inputs
    cifar = True
    if x is None:
        cifar = False
        ## For MNIST (default) return placeholders for feeding dictionaries
        x = tf.placeholder(tf.float32, shape=[None, 784], name='X')

    # Hidden layers
    previous_layer = x
    for layer_index, layer_dim in enumerate(architecture):
        previous_layer = build_layer(previous_layer, layer_dim, 'layer_{}'.format(layer_index), **kwargs)

    # Output layer
    logits = build_layer(previous_layer, 10, 'layer_{}'.format(len(architecture)), activation_fn='linear',
                         initializer=kwargs['initializer'])

    probabilities = tf.nn.softmax(logits, name="softmax_tensor")
    if y is None:
        # The labels are 1-hot encoded for MNIST
        y = tf.placeholder(tf.float32, shape=[None, 10], name='y')
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y, axis=1), logits=logits), name='loss')
        correct_predictions = tf.equal(tf.argmax(probabilities, 1), tf.argmax(y, 1))
    else:
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits), name='loss')
        correct_predictions = tf.equal(tf.cast(tf.argmax(probabilities, 1), tf.int32), y)
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name='accuracy')

    if not cifar:
        # CIFAR doesn't logging at this level
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('accuracy', accuracy)

    metrics = {
        'probabilities': probabilities,
        'loss': loss,
        'accuracy': accuracy,
        'logits': logits
        }

    placeholder = {'x': x, 'y': y}
    return placeholder, metrics

def build_deep_nn_model_w_placeholders(architecture, placeholders, graph=tf.get_default_graph(), x=None, y=None, **kwargs):
    """Build a deep fully connected NN model. kwargs are passed to build layer function."""
    # Inputs
    # Hidden layers
    previous_layer = placeholders['x']
    for layer_index, layer_dim in enumerate(architecture):
        previous_layer = build_layer(previous_layer, layer_dim, 'layer_{}'.format(layer_index), **kwargs)

    # Output layer
    logits = build_layer(previous_layer, 10, 'layer_{}'.format(len(architecture)), activation_fn='linear',
                         initializer=kwargs['initializer'])

    probabilities = tf.nn.softmax(logits, name="softmax_tensor")

    # The labels are 1-hot encoded for MNIST
    y = placeholders['y']
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y, axis=1), logits=logits), name='loss')
    correct_predictions = tf.equal(tf.argmax(probabilities, 1), tf.argmax(y, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name='accuracy')

    metrics = {
        'probabilities': probabilities,
        'loss': loss,
        'accuracy': accuracy,
        'logits': logits
        }
    return placeholders, metrics
