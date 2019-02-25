import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


df = pd.read_csv(sys.argv[1], index_col=0)

def format_run(series):
    if series['model.dropout_rate'] == 0.5:
        return 'dropout'
    elif series['model.batch_normalization'] is True:
        return 'batch normalization'
    else:
        return 'vanilla'

df['architecture'] = df.apply(format_run, axis=1)

# remove config values that were not changed across runs
n_uniue_values = {col: len(pd.unique(df[col])) for col in df.columns}
drop_cols = [col for col, n_unique in n_uniue_values.items() if n_unique == 1]
df.drop(drop_cols, axis=1, inplace=True)
# Melt performance measured
melted = df.melt(id_vars=['architecture'], value_vars=['step', 'test_acc', 'total_persistence_normalized', 'val_acc', 'val_loss'])

g = sns.FacetGrid(melted, col='variable', hue='architecture', col_order=['total_persistence_normalized','test_acc'], margin_titles=True, sharey=False, sharex=False)
g.map(sns.distplot, 'value')
g.add_legend()
plt.savefig(sys.argv[2])
