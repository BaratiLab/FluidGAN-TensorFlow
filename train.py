import tensorflow as tf
import os
import numpy as np
import time

from datasets.data_loader import HDF5Loader
from models.cGAN_model import CGANModel
from options.options import TrainOptions
from util.check import CheckOp
from util.transform import list2tensor
from util.fetch import TrainFetch, should, show_result_train
from util.summary import add_summaries

def main():

    args = TrainOptions().initialize().parse_args()
    args = CheckOp(args).check_main() 

    loader = HDF5Loader(args)
    inputs, targets = loader.get_item()
    # create model
    model = CGANModel(args, inputs, targets)
    # add initial checkpoint
    init_op = tf.global_variables_initializer()
    sum_ops = add_summaries(args, model)
    # define saver for saving and restoring
    # all variables must be in front of this line!
    saver = tf.train.Saver(max_to_keep=2**20)

    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter(args.summary_path, sess.graph)
        start_time = time.time()
        with loader.loader.begin(sess):
            sess.run(init_op)
            for step in range(args.max_step):            
                fetches = TrainFetch(model, args, step, start_time)
                results = sess.run(fetches.content)
                show_result_train(args, results, step, start_time)
                if should(args.summary_freq, step, args.max_step):
                    s_val = sess.run(sum_ops) 
                    summary_writer.add_summary(s_val, global_step=step) 
                if should(args.save_freq, step, args.max_step):
                    print("Saving model...")
                    saver.save(sess, args.save_path + '/model', global_step=step) 
                    
            # End loader session
            os._exit(0)
if __name__ == "__main__":       
    main()