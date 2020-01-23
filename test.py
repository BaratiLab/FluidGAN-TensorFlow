import tensorflow as tf
import os
import numpy as np
import h5py

from datasets.data_loader import HDF5Loader
from models.cGAN_model import CGANModel
from options.options import TestOptions
from util.check import CheckOp
from util.fetch import TestFetch, should, show_result_test
from util.save import create_dataset, save_h5

def main():

    args = TestOptions().initialize().parse_args()
    args = CheckOp(args).check_main() 

    loader = HDF5Loader(args)
    inputs, targets = loader.get_item()
    model = CGANModel(args, inputs, targets)

    saver = tf.train.Saver() # must have this line even saver is not used!
    new_saver = tf.train.import_meta_graph(args.model_path)

    with tf.Session() as sess:
        #sess.run(init_op)
        new_saver.restore(sess, os.path.splitext(args.model_path)[0])
        with h5py.File(args.result_path, 'w') as f:
            create_dataset(args, f, model.pred_shape)
            with loader.loader.begin(sess):
                for step in range(args.max_step):
                    fetches = TestFetch(model, args, step)
                    results = sess.run(fetches.content)
                    show_result_test(args, results, step)
                    # save results for every step
                    save_h5(args, f, results, step)
                    if should(args.progress_freq, step, args.max_step):
                        print("step %d / %d" %(step, args.max_step))
                
    #         # End loader session
                os._exit(0)
if __name__ == "__main__":       
    main()