import os
import pandas as pd
import json
from argparse import ArgumentParser


def compress_dict(dictionary, prefix=''):
    output = {}
    for key, value in dictionary.items():
        if type(value) is dict:
            output.update(compress_dict(value, prefix=key+'.'))
        elif type(value) is list:
            output[prefix+key] = str(value)
        else:
            output[prefix+key] = value
    return output


def get_config_dictionary(filename):
    with open(filename) as f:
        data = json.load(f)

    return compress_dict(data)


def remove_keys_from_dict(dictionary, keys):
    for key in keys:
        if key in dictionary.keys():
            del dictionary[key]
    return dictionary


def get_results_dictionary(filename):
    with open(filename) as f:
        data = json.load(f)

    if data['status'] != 'COMPLETED':
        print('Run {} not completed and will be ignored'.format(filename))
        return None

    result = remove_keys_from_dict(data['result'], ['default_factory', 'py/object'])

    return result


def repeat_values(dictionary, repeats):
    for key, value in dictionary.items():
        dictionary[key] = [value]*repeats
    return dictionary


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('run_dir', nargs='+', help='Directories containing sacred config.json and run.json files')
    parser.add_argument('--output', required=True, help='Filename containing combined runs')
    args = parser.parse_args()
    run_outputs = []
    for run_dir in args.run_dir:
        try:
            config = get_config_dictionary(os.path.join(run_dir, 'config.json'))
        except FileNotFoundError:
            print('{} does not seem to be a sacred run'.format(run_dir))
            continue
        results = get_results_dictionary(os.path.join(run_dir, 'run.json'))
        if results is None:
            continue
        config = repeat_values(config, config['run.runs'])
        joined = config.copy()
        joined.update(results)
        run_outputs.append(pd.DataFrame.from_dict(joined))
    all_runs = pd.concat(run_outputs)
    all_runs.to_csv(args.output)


