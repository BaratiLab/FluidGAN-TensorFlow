import math
from matplotlib import pyplot as plt 
from matplotlib import cm
import os

def plot_display(args, results, step):
    data_name_dic = parse_display(args, results, step)
    plot_coutour(args, data_name_dic)
def plot_display_verbose(args, results, step):
    sample_name_dic, full_name_dic = parse_display_verbose(args, results, step)
    plot_pixel(args, sample_name_dic, full_name_dic)


def parse_display_verbose(args, results, step):
    cur_epoch = math.ceil(step / args.steps_per_epoch)
    sample_name_dic = {}
    full_name_dic = {}
    for cata in ["gen_all", "dis_fake_all", "dis_real_all"]:
        all_data_list = results[cata]
        for idx, one_data in enumerate(all_data_list):
            one_data = one_data[0] # (2 * 2 * 512) like
            one_sample = one_data[:, :, -1] # a square plot
            one_full = one_data.reshape(-1, one_data.shape[-1])
            one_name_sample = "%d_%d_%s_l%d_s.png" %(cur_epoch, step, cata, idx+1)
            one_name_full = "%d_%d_%s_l%d_f.png" %(cur_epoch, step, cata, idx+1)
            sample_name_dic[one_name_sample] = one_sample
            full_name_dic[one_name_full] = one_full
    return sample_name_dic, full_name_dic
def parse_display(args, results, step):
    cur_epoch = math.ceil(step / args.steps_per_epoch)
    data_name_dic = {}
    ch_name_dic = {0:"u", 1:"v", 2:"p", 3:"T"}
    for cata in ["inputs", "targets", "outputs"]:
        all_data = results[cata][0]
        for ch in ch_name_dic.keys():
            one_data = all_data[:, :, ch]
            one_name = "%d_%d_%s_%s.png" %(cur_epoch, step, cata, ch_name_dic[ch])
            data_name_dic[one_name] = one_data
    return data_name_dic
def plot_coutour(args, data_name_dic):
    ch_name_dic = {"u":0, "v":1, "p":2, "T":3}
    for one_name in data_name_dic.keys():
        ch = one_name.split('_')[-1]
        ch = ch.split('.')[0]
        ch_idx = ch_name_dic[ch]
        one_data = data_name_dic[one_name]
        max_value = args.max_value[ch_idx]
        min_value = args.min_value[ch_idx]
        fig = plt.gcf()
        fig.set_size_inches(15/3, 15/3)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.contourf(one_data, 50, cmap=cm.jet, vmin=min_value, vmax=max_value)
        plt.savefig(os.path.join(args.display_result_path, one_name))
        plt.close()
def plot_pixel(args, sample_name_dic, full_name_dic):
    for one_name in sample_name_dic.keys():
        fig = plt.gcf()
        fig.set_size_inches(15/3, 15/3)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.imshow(sample_name_dic[one_name], cmap=cm.jet)
        plt.savefig(os.path.join(args.display_verbose_path, one_name))
        plt.close()
    for one_name in full_name_dic.keys():
        fig = plt.gcf()
        plt.margins(0, 0)
        plt.imshow(full_name_dic[one_name], cmap=cm.jet)
        plt.xlabel("channel")
        plt.ylabel("pixel")
        plt.savefig(os.path.join(args.display_verbose_path, one_name))
        plt.close()


