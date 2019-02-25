import os
import tensorflow as tf
from sacred.observers import FileStorageObserver

from src.batchnorm_simple import ex as batchnorm_experiment
from src.dropout_simple import ex as dropout_experiment
filepath = os.path.dirname(os.path.abspath(__file__))

# Store results of runs in results/runs/exeriment
# testing parameters:
# run_parameters = {'runs': 2, 'epochs': 1}
run_parameters = {}
filestorage_path = os.path.join(filepath, "results", "runs")
batchnorm_experiment.observers.append(FileStorageObserver.create(filestorage_path))
dropout_experiment.observers.append(FileStorageObserver.create(filestorage_path))

for dropout_rate in [0.0, 0.5]:
    dropout_experiment.run(config_updates={'model.dropout_rate': dropout_rate, 'run': run_parameters})
    tf.reset_default_graph()

batchnorm_experiment.run(config_updates={'model.batch_normalization': True, 'run': run_parameters})
