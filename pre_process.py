from datasets.dataset import SteadyStateDataset, BigTimeDataset
from options.options import  PreProcessOptions
import os


def main():
    args = PreProcessOptions().initialize().parse_args()
    args = get_path(args)
    dataset = model_selection(args)
    if not os.path.exists(args.maxmin_path):      
        dataset.compute_maxmin(args.maxmin_path)
    if args.movie != 1:
        dataset.shuffle_data(args.shuffle_path, args.num_chunk)
    dataset.make_sample(args.num_sample, args.shuffle_path, args.sample_dir)

def get_path(args):
    base = os.getcwd()
    args.raw_path = os.path.join(base, "datasets/raw/%s.h5" %args.name)
    args.maxmin_path = os.path.join(base, "datasets/raw/%s.csv" %args.name)
    args.shuffle_path = os.path.join(base, "datasets/shuffle/%s.h5" %args.name)
    args.sample_dir = os.path.join(base, "datasets/sample/%s" %args.name)
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)
    return args

def model_selection(args):
    if args.mode == "steady":
        dataset = SteadyStateDataset(args)
        assert args.movie == None
    elif args.mode == "big_time":
        dataset = BigTimeDataset(args)
        assert args.movie != None
        assert args.num_chunk != None
        #assert args.num_chunk * 5 < args.num_sample
    return dataset

if __name__ == "__main__":
    main()