#!/usr/bin/env python
"""
Create ETL-processed CSVs from an SQLite3 database generated with bfaut

Usage:
    ./etl.py [--debug] [--sqlite=<path>]
    ./etl.py -h|--help
    ./etl.py -v|--version

Options:
    -h, --help          Print help and exit
    -v, --version       Print version and exit
    --debug             Execute a command with debug messages
    --sqlite=<path>     Set a path to an SQLite3 database
                        [default: lightning.sqlite3]

Example:
    Target databases are generated with bfaut (github.com/dceoy/bfaut).

    $ bfaut stream --sqlite=lightning.sqlite3 \
          lightning_executions_FX_BTC_JPY lightning_ticker_FX_BTC_JPY
    $ ./etl.py --sqlite=lightning.sqlite3
"""

import logging
import os
import sqlite3
from docopt import docopt
import numpy as np
import pandas as pd
import seaborn as sns

__version__ = 'v0.0.1'


def main():
    args = docopt(__doc__, version='./etl.py {}'.format(__version__))
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=(logging.DEBUG if args['--debug'] else logging.WARNING)
    )
    logging.debug('args:{0}{1}'.format(os.linesep, args))

    with sqlite3.connect('lightning.sqlite3') as con:
        dfs = {
            csv: pd.read_sql(sql, con=con) for csv, sql
            in {
                'exec':
                'SELECT exec_date, price, side, size FROM '
                'lightning_executions_FX_BTC_JPY;',
                'tick':
                'SELECT timestamp, ltp FROM lightning_ticker_FX_BTC_JPY;'
            }.items()
        }
        logging.debug('dfs:{0}{1}'.format(os.linesep, dfs))

    for k in ['tick', 'exec']:
        dfs[k].to_csv('df_{}.csv'.format(k), index=False)

    df_exec_delta = dfs['exec'].assign(
        signed_size=lambda d:
        np.where(d['side'] == 'BUY', d['size'], - d['size'])
    ).rename(
        columns={'exec_date': 'timestamp'}
    ).groupby('timestamp')['signed_size'].sum().to_frame(name='size')
    df_exec_delta.to_csv('df_exec_delta.csv', index=True)

    alpha = [0.01, 0.05, 0.1]
    df_ewm = pd.concat([
        df_exec_delta.reset_index().assign(
            alpha=a,
            ewma=lambda d: d['size'].ewm(com=(1 / a - 1)).mean(),
            ewmstd=lambda d: d['size'].ewm(com=(1 / a - 1)).std()
        ) for a in alpha
    ]).dropna(how='any')
    df_ewm.to_csv('df_ewm.csv', index=False)

    sns.set(style='darkgrid', color_codes=True)
    g = sns.FacetGrid(
        df_ewm, col='alpha', hue='alpha', row_order=alpha, col_wrap=3
    )
    g = g.map(sns.pointplot, 'timestamp', 'ewma', marker='.')
    g.savefig('ewma.png')


def load_dfs(dir_path='.'):
    return {
        k: pd.read_csv(
            os.path.join(dir_path, 'df_{}.csv'.format(k)),
            index_col='timestamp', parse_dates=True
        )
        for k in ['tick', 'exec']
    }


if __name__ == '__main__':
    main()
