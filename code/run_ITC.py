import argparse

from openea.modules.args.args_hander import load_args
from data_model import DataModel
from predicate_alignment import PredicateAlignModel
from MultiKE_CSL import MultiKE_CV


parser = argparse.ArgumentParser(description='run')
parser.add_argument('--training_data', type=str, default='')
parser_args = parser.parse_args()


if __name__ == '__main__':
    args = load_args('args.json')
    args.training_data = parser_args.training_data
    data = DataModel(args)
    attr_align_model = PredicateAlignModel(data.kgs, args)
    model = MultiKE_CV(data, args, attr_align_model)
    model.run()

