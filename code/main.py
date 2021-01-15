import os
import argparse

from pytorch.utils import load_args
from pytorch.run import run_itc, run_ssl


parser = argparse.ArgumentParser(description='main')
parser.add_argument('-m', '--mode', type=str, required=True)
parser.add_argument('-d', '--data', type=str, required=True)
parser_args = parser.parse_args()


if __name__ == '__main__':
    args = load_args(os.path.join(os.path.dirname(__file__), 'pytorch', 'args.json'))
    args.training_data = parser_args.data

    if parser_args.mode == 'ITC':
        run_itc(args)
    elif parser_args.mode == 'SSL':
        run_ssl(args)
