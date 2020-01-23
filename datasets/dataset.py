import numpy as np
import h5py
import os 
from tqdm import tqdm
import random
import sys
sys.path.append("..")
#from util.plot import plot_dataset

def dir_check(dir_root):
    if not os.path.exists(dir_root):
        os.makedirs(dir_root)

class SteadyStateDataset():

    def __init__(self, args):
        self.data_path = args.raw_path
        self.dataset_name = args.name
        self.data_check()

    def data_check(self):
        with h5py.File(self.data_path, "r") as f:
            key_list = [key for key in f.keys()]
            assert len(key_list) == 1
            assert key_list[0] == "data"
            print("Number of sample is %d\tshape is" %f["data"].shape[0], f["data"].shape)

    def compute_maxmin(self, maxmin_path):
        with h5py.File(self.data_path, "r") as f:
            max_para = f["data"].value.max(axis=(0,1,2))
            min_para = f["data"].value.min(axis=(0,1,2))
            ch_n = int(max_para.shape[0] / 2)
            c_max_para = np.zeros(ch_n)
            c_min_para = np.zeros(ch_n)
            for ch in range(ch_n):
                c_max_para[ch] = max(max_para[ch], max_para[ch + ch_n])
                c_min_para[ch] = min(min_para[ch], min_para[ch + ch_n])
                para = np.concatenate((c_max_para, c_min_para), axis=0)
                np.savetxt(maxmin_path, para, delimiter = ',') 
        self.maxmin_path = maxmin_path
        print("Saving maxmin complete")

    def shuffle_data(self, shuffle_path):
        with h5py.File(self.data_path, "r") as sf:
            with h5py.File(shuffle_path, "w") as df:
                data = sf["data"].value
                np.random.shuffle(data)
                df.create_dataset("data", data=data)
        print("Shuffling data complete")
        self.shuffle_path = shuffle_path

    def make_sample(self, num_sample, sample_dir, num_fold=5): 
        self.sample_dir = sample_dir
        dir_check(self.sample_dir)
        self.num_fold = num_fold
        self.num_sample = num_sample
        self.split_shuffle(num_sample)
        self.merge()

    def merge(self):
        with h5py.File(os.path.join(self.sample_dir, "1.h5"), 'r') as f:
            shape = f["data"].shape
        for i in range(self.num_fold):
            train_data = np.zeros((shape[0] * (self.num_fold - 1), shape[1], shape[2], shape[3]))
            train_path = os.path.join(self.sample_dir, "%d_train_f%d.h5" %(self.num_sample, (i+1)))
            test_path = os.path.join(self.sample_dir, "%d_test_f%d.h5" %(self.num_sample, (i+1)))
            count = 0
            for j in range(self.num_fold):
                if j == i:
                    os.chdir(self.sample_dir)
                    os.system("cp %d.h5 %s" %(j + 1, test_path))
                else:
                    with h5py.File(os.path.join(self.sample_dir, "%d.h5"%(j+1)), 'r') as of:
                        train_data[count: count + shape[0], :, :, :] = of["data"].value
                        count += shape[0]
            with h5py.File(train_path, 'w') as tf:
                tf.create_dataset("data", data=train_data)
        for i in range(self.num_fold):
            os.chdir(self.sample_dir)
            os.system("rm %d.h5" %(i+1))
        print("Making data complete!")

    def split_shuffle(self, num_sample):
        num_fold_sample = int(num_sample / self.num_fold)
        with h5py.File(self.shuffle_path, 'r') as sf:
            count = 0
            for fold in range(self.num_fold):
                one_h5_path = os.path.join(self.sample_dir, "%d.h5" %(fold + 1))
                with h5py.File(one_h5_path, 'w') as of:
                    of.create_dataset("data", data=sf["data"].value[count: count + num_fold_sample])
        print("Spliting dataset complete")
            
class BigTimeDataset():
    
    def __init__(self, args):
        self.data_path = args.raw_path
        self.dataset_name = args.name
        self.data_check()

    def data_check(self):
        with h5py.File(self.data_path, "r") as f:
            key_list = [key for key in f.keys()]
            self.num_movie = len(key_list)
            self.num_frame = f[key_list[0]].shape[0]
            self.shape = f[key_list[0]].shape
            print("The number of movies is %d" %len(key_list))
            print("Each movie has %d frames\tshape is" %f[key_list[0]].shape[0], f[key_list[0]].shape)
            
    def compute_maxmin(self, maxmin_path):
        with h5py.File(self.data_path, "r") as f:
            for idx, key in tqdm(enumerate(f.keys())):
                if idx == 0:
                    max_para = f[key].value.max(axis=(0,1,2))
                    min_para = f[key].value.min(axis=(0,1,2))
                else:
                    new_max = f[key].value.max(axis=(0,1,2))
                    new_min = f[key].value.min(axis=(0,1,2))
                    for ch in range(max_para.shape[0]):
                        if new_max[ch] > max_para[ch]:
                            max_para[ch] = new_max[ch]
                        if new_min[ch] < min_para[ch]:
                            min_para[ch] = new_min[ch]   
            ch_n = int(max_para.shape[0] / 2)
            c_max_para = np.zeros(ch_n)
            c_min_para = np.zeros(ch_n)
            for ch in range(ch_n):
                c_max_para[ch] = max(max_para[ch], max_para[ch + ch_n])
                c_min_para[ch] = min(min_para[ch], min_para[ch + ch_n])
                para = np.concatenate((c_max_para, c_min_para), axis=0)
                np.savetxt(maxmin_path, para, delimiter = ',') 
    
    def shuffle_data(self, shuffle_path, num_chunk):
        with h5py.File(shuffle_path, "w") as sf:
            with h5py.File(self.data_path, "r") as rf:
                key_list = [key for key in rf.keys()]
                shuffled_data = np.zeros((num_chunk, self.shape[1], self.shape[2], self.shape[3]))
                random.shuffle(key_list)
                count = 0
                id_chunk = 0
                for inx, key in tqdm(enumerate(key_list)):
                    assert self.num_frame < num_chunk
                    if self.num_frame > num_chunk - count: 
                        temp = np.zeros((count, self.shape[1], self.shape[2], self.shape[3]))
                        temp[:, :, :, :] = shuffled_data[0: count, :, :, :]
                        np.random.shuffle(temp)
                        sf.create_dataset("data{}".format(id_chunk), data=temp)
                        count = 0
                        id_chunk += 1
                        print("Filling chunk {} complete".format(id_chunk))
                        shuffled_data = np.zeros((num_chunk, self.shape[1], self.shape[2], self.shape[3]))
                    else:
                        shuffled_data[count: count + self.num_frame, :, :, :] = rf[key].value
                        count += self.num_frame

    def make_sample(self, num_sample, shuffle_path, sample_dir, num_fold=5):
        self.make_all(num_sample, shuffle_path, sample_dir)
        self.split(num_sample, num_fold, sample_dir)
        self.merge(num_sample, num_fold, sample_dir)

    def make_all(self, num_sample, shuffle_path, sample_dir):
        with h5py.File(shuffle_path, "r") as sf:
            key_list = []
            key_file_dic = {}
            for key in sf.keys():           
                key_list.append(key)  
                key_file_dic[key] = sf[key].shape[0]  
            print("Creating sample with number %d..."%(num_sample))
            with h5py.File(os.path.join(sample_dir, "all.h5"), "w") as df:
                for keyid, key in enumerate(key_list):
                    cur_num = key_file_dic[key]
                    if num_sample < cur_num:
                        temp = np.zeros((num_sample, self.shape[1], self.shape[2], self.shape[3]))
                        temp[:, :, :, :] = sf[key][:num_sample, :, :, :]
                        df.create_dataset(key, data=temp)
                        print("Filling chunk %d with number %d, complete"%(keyid + 1, num_sample))
                        break
                    else:
                        df.create_dataset(key, data=sf[key].value)
                        print("Filling chunk %d with number %d, complete"%(keyid + 1, cur_num))
                        num_sample -= key_file_dic[key]
                        print("%d files remaining" %(num_sample))
    
    def split(self, num_sample, num_fold, sample_dir):
        one_fold = int(num_sample / num_fold)
        with h5py.File(os.path.join(sample_dir, "all.h5"), "r") as sf:
            key_list = []
            key_file_dic = {}
            for key in sf.keys():           
                key_list.append(key)  
                key_file_dic[key] = sf[key].shape[0]  
            for fold_id in range(num_fold):
                one_name = os.path.join(sample_dir, "fold%d.h5" %(fold_id+1))
                print("Creating fold %d" %fold_id)
                with h5py.File(one_name, 'w') as df:
                    df.create_dataset("data", shape=(one_fold, self.shape[1], self.shape[2], self.shape[3]))
                    start_count = one_fold * fold_id
                    end_count = one_fold * (fold_id + 1)
                    print("Start count is %d" %start_count)
                    print("End count is %d" %end_count)

                    total_count = 0
                    for key_ind, key in enumerate(key_list):
                        cur_num = key_file_dic[key]
                        if total_count <= start_count and total_count + cur_num > start_count:
                            start_key = key_ind
                            start_remain = start_count - total_count 
                            print("Start at key index %d, %d files remaining" % (start_key, start_remain))
                        if total_count < end_count and total_count + cur_num >= end_count:
                            end_key = key_ind
                            end_remain = end_count - total_count
                            print("End at key index %d, %d files remaining" % (end_key, end_remain))
                        total_count += cur_num
                                    
                    fold_count = 0
                    for key_ind in range(start_key, end_key + 1):
                        key = key_list[key_ind]
                        cur_num = key_file_dic[key]
                        print("Filling key index %d" %(key_ind))
                        key = key_list[key_ind]
                        print("key name: %s, key index: %d, file number: %d" %(key, key_ind, cur_num))
                        print("Fold count: %d" %fold_count)
                        if key_ind == start_key and start_remain != 0:
                            df["data"][fold_count: fold_count + cur_num - start_remain, :, :, :] = sf[key][start_remain:, :, :, :]
                            fold_count += (cur_num - start_remain)
                        elif key_ind == end_key and end_remain != 0 :
                            df["data"][fold_count: fold_count + end_remain, :, :, :] = sf[key][:end_remain, :, :, :] 
                            fold_count += end_remain
                        else:
                            df["data"][fold_count: fold_count + cur_num, :, :, :] = sf[key][:, :, :, :]
                            fold_count += cur_num

    def merge(self, num_sample, num_fold, sample_dir):
        print("Merging training data..")
        one_fold = int(num_sample / num_fold)
        for fold_id in range(num_fold):
            print("Merging training fold %d" % (fold_id+1))
            train_path = os.path.join(sample_dir, "%d_f%d_train.h5" % (num_sample, (fold_id+1)))
            train_list = []
            for i in range(num_fold):
                # test data
                if i == fold_id:
                    fold_path = os.path.join(sample_dir, "fold%d.h5" % (i+1))
                    test_path = os.path.join(sample_dir, "%d_f%d_test.h5" %(num_sample, (fold_id+1)))
                    os.system("cp %s %s" %(fold_path, test_path))
                if i != fold_id:
                    train_list.append(os.path.join(sample_dir, "fold%d.h5" % (i+1)))

            with h5py.File(train_path, 'w') as df:
                df.create_dataset("data", shape=(one_fold * (num_fold-1), self.shape[1], self.shape[2], self.shape[3]))
                count = 0
                for src in tqdm(train_list):
                    with h5py.File(src, 'r') as sf:
                        temp_shape = sf["data"].shape
                        df["data"][count: count + temp_shape[0], :, :, :] = sf["data"][:, :, :, :]
                        count += temp_shape[0]

