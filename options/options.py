import argparse
class BaseOptions():
    """ options used during training and testing 

    """
    def initialize(self):
        """ initialize body """
        parser = argparse.ArgumentParser()
        # path 
        parser.add_argument("--input_path", type=str, required=True, help="path of hdf5 input file")
        parser.add_argument("--output_path", type=str, required=True, help="path of output directory")
        parser.add_argument("--maxmin_path", type=str, required=True, help="csv file for max and min")
        # model params
        parser.add_argument("--ngf", type=int, default=64, help="number of generator filters in first conv layer")
        parser.add_argument("--ndf", type=int, default=64, help="number of discriminator filters in first conv layer")
        # other params 
        parser.add_argument("--data_name", type=str, required=True)
        parser.add_argument("--dataset_name", default='/data', type=str, help="inner hdf5 dataset path")
        parser.add_argument("--num_sample", type=int)
        parser.add_argument("--steps_per_epoch", type=int)            
        parser.add_argument("--max_step", type=int)    
        parser.add_argument("--width", type=int)    
        parser.add_argument("--num_ch", type=int)  
        parser.add_argument("--max_value", type=float)
        parser.add_argument("--min_value", type=float)
        parser.add_argument("--display_verbose", type=int, default=0)
        parser.add_argument("--progress_freq", type=int, required=True)
        parser.add_argument("--display_freq", type=int, required=True)
        return parser

class TrainOptions():
    def __init__(self):
        self.base_options = BaseOptions()
        self.base_parser = self.base_options.initialize()
    def initialize(self):
        parser = self.base_parser
        # hypers
        parser.add_argument("--batch_size", type=int, required=True)
        parser.add_argument("--max_epoch", type=int, required=True)       
        parser.add_argument("--disc_lr", type=float, default=0.00002, help="initial learning rate for adam")
        parser.add_argument("--disc_beta1", type=float, default=0.5, help="momentum term of adam")
        parser.add_argument("--gen_lr", type=float, default=0.0002, help="initial learning rate for adam")
        parser.add_argument("--gen_beta1", type=float, default=0.5, help="momentum term of adam")
        parser.add_argument("--l1_weight", type=float, default=100.0, help="weight on L1 term for generator gradient")
        parser.add_argument("--gan_weight", type=float, default=1.0, help="weight on GAN term for generator gradient")
        
        # model params
        parser.add_argument("--eps", type=float, default=1e-12, help="precision")

        # opt
        parser.add_argument("--mode", type=str, default="train")   
        parser.add_argument("--summary_freq", type=int, required=True)
        parser.add_argument("--save_freq", type=int, required=True)        
        return parser        

class TestOptions():
    def __init__(self):
        self.base_options = BaseOptions()
        self.base_parser = self.base_options.initialize()
    def initialize(self):
        parser = self.base_parser
        # hypers
        parser.add_argument("--batch_size", type=int, default=1)
        parser.add_argument("--max_epoch", type=int, default=1)
        parser.add_argument("--num_sample", type=int, required=True)
        # path
        parser.add_argument("--checkpoint", type=str, required=True)
        # opt
        parser.add_argument("--mode", type=str, default="test")   
        parser.add_argument("--restore_step", type=int, required=True)
        return parser

class PreProcessOptions():
    def initialize(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--mode", type=str, required=True)
        parser.add_argument("--movie", type=int)
        parser.add_argument("--name", type=str, required=True)
        parser.add_argument("--num_sample", type=int, required=True)
        parser.add_argument("--num_chunk", type=int)
        return parser

