import os
import h5py
import math
import numpy as np

class CheckOp():

	def __init__(self, args):
		self.args = args

	def check_path(self, path):
		if not os.path.exists(path):
			os.makedirs(path)

	def add_path(self):
		# display
		self.args.output_path += "/%s" %self.args.data_name
		if self.args.mode == "train":
			self.args.summary_path = os.path.join(self.args.output_path, "log")
			self.args.save_path = os.path.join(self.args.output_path, "save")			
		else:
			self.args.model_path = os.path.join(self.args.checkpoint, "model-%d.meta"%(self.args.restore_step-1))
			self.args.result_path = os.path.join(self.args.output_path, "results.h5") 
		self.args.display_result_path = os.path.join(self.args.output_path, "display/result")
		if self.args.display_verbose:
			self.args.display_verbose_path = os.path.join(self.args.output_path, "display/verbose")	
	def make_path(self):
		self.add_path()

		self.check_path(self.args.output_path)
		self.check_path(self.args.display_result_path)
		if self.args.display_verbose:
			self.check_path(self.args.display_verbose_path)
		if self.args.mode == "train":

			self.check_path(self.args.summary_path)
			self.check_path(self.args.save_path)


	def print_args(self):
		for k, v in self.args._get_kwargs():
			print(k, "=", v)
		print("=" * 50) 

	def get_number(self):
		with h5py.File(self.args.input_path, "r") as f:
			shape = f[self.args.dataset_name].shape
			self.args.num_sample = shape[0]
			self.args.width = shape[1]
			assert shape[1] == shape[2]
			self.args.num_ch = int(shape[3] / 2)
			self.args.steps_per_epoch = int(math.ceil(self.args.num_sample / self.args.batch_size))
			self.args.max_step = self.args.steps_per_epoch * self.args.max_epoch
			print("examples count: %d" % self.args.num_sample)
			print("steps per epoch: %d" % self.args.steps_per_epoch)
	
	def get_maxmin(self):
		para = np.loadtxt(open(self.args.maxmin_path, 'rb'), delimiter = ',')
		self.args.max_value = [para[i] for i in range(self.args.num_ch)]
		self.args.min_value = [para[i] for i in range(self.args.num_ch, 2 * self.args.num_ch)]

	def check_main(self):
		self.get_number()
		assert self.args.num_sample % self.args.batch_size == 0
		self.get_maxmin()
		self.make_path()
		self.print_args()
		return self.args

