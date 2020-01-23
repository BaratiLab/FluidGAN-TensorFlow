from .transform import upscale
from .plot import plot_display, plot_display_verbose
import tensorflow as tf
import time
import math

def should(freq, step, max_step):
	return freq > 0 and ((step + 1) % freq == 0 or step == max_step - 1)

def show_result_train(args, results, step, start_time):
    if should(args.progress_freq, step, args.max_step):
        show_progress_result(args, results, step, start_time)
    if should(args.display_freq, step, args.max_step):
        show_display_result(args, results, step)

def show_result_test(args, results, step):
    if should(args.display_freq, step, args.max_step):
        show_display_result(args, results, step)

def show_progress_result(args, results, step, start_time):
    rate = (step + 1) * args.batch_size / (time.time() - start_time)
    remaining = (args.max_step - step) * args.batch_size / rate
    epoch_num = math.ceil(step / args.steps_per_epoch)
    print("progress epoch %d step %d  image/sec %0.1f  remaining %dm" % (epoch_num, step, rate, remaining / 60))
    print("discrim_loss", results["discrim_loss"])
    print("gen_loss_GAN", results["gen_loss_GAN"])
    print("gen_loss_L1", results["gen_loss_L1"])

def show_display_result(args, results, step):
    plot_display(args, results, step)
    if args.display_verbose:
        plot_display_verbose(args, results, step)
class TestFetch():
    def __init__(self, model, args, step):
        self.model = model
        self.args = args
        self.step = step
        self.content = {}
        self.add_result()
        if should(args.display_freq, step, args.max_step):
            self.add_verbose()
    def add_result(self):
        self.content["inputs"] = upscale(self.args, self.model.inputs)
        self.content["targets"] = upscale(self.args, self.model.targets)
        self.content["outputs"] = upscale(self.args, self.model.outputs)
        self.content["predict_fake"] = self.model.predict_fake_map
        self.content["predict_real"] = self.model.predict_real_map
    def add_verbose(self):
        if self.args.display_verbose:
            self.content["gen_all"] = self.model.all_g_layers
            self.content["dis_fake_all"] = self.model.all_d_fake_layers
            self.content["dis_real_all"] = self.model.all_d_real_layers
    
class TrainFetch():

    def __init__(self, model, args, step, start_time):
        self.model = model
        self.args = args
        self.step = step
        self.content = {}
        self.start_time = start_time
        self.add_train()
        if should(args.progress_freq, step, args.max_step):
            self.add_progress()
        if should(args.display_freq, step, args.max_step):
            self.add_display()

    def add_train(self):
        self.content["train"] = self.model.train_op
    
    def add_progress(self):
        self.content["discrim_loss"] = self.model.discrim_loss
        self.content["gen_loss_GAN"] = self.model.gen_loss_GAN
        self.content["gen_loss_L1"] = self.model.gen_loss_L1  

    def add_display(self):
        self.content["inputs"] = upscale(self.args, self.model.inputs)
        self.content["targets"] = upscale(self.args, self.model.targets)
        self.content["outputs"] = upscale(self.args, self.model.outputs)
        if self.args.display_verbose:
            self.content["gen_all"] = self.model.all_g_layers
            self.content["dis_fake_all"] = self.model.all_d_fake_layers
            self.content["dis_real_all"] = self.model.all_d_real_layers

