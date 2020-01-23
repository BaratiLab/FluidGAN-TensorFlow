import h5py
import tftables
import tensorflow as tf
import numpy as np
import sys
sys.path.append("..")
from util.transform import list2tensor

class HDF5Loader(object):
    def __init__(self, args):
        self.args = args
        self.loader_init()
    def loader_init(self):
        with tf.device('/cpu:0'):
            ordered = True if self.args.mode == "train" else False
            self.loader = tftables.load_dataset(filename=self.args.input_path,
                                        dataset_path=self.args.dataset_name,
                                        input_transform=self.transform,
                                        batch_size=self.args.batch_size,
                                        ordered=ordered)
    def transform(self, one_batch):
        inputs = tf.cast(one_batch[:, :, :, :self.args.num_ch], tf.float32)
        targets = tf.cast(one_batch[:, :, :, self.args.num_ch:], tf.float32)
        max_value, min_value = list2tensor(self.args)
        # scale here
        inputs = (inputs - min_value) / (max_value - min_value)
        targets = (targets - min_value) / (max_value - min_value)
        return inputs, targets       
    def get_item(self):
        # inputs and targets are [batch_size, height, width, channels]
        inputs, targets = self.loader.dequeue()
        return inputs, targets
    
        