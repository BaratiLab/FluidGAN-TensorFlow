import h5py

def save_h5(args, f, results, step):

    for key in ["inputs", "outputs", "targets", "predict_fake", "predict_real"]:
        one_data = results[key]
        if step < args.max_step - 1:
            f[key][step * args.batch_size: (step+1) * args.batch_size, :, :, :] = one_data
        else:
            f[key][step * args.batch_size:, :, :, :] = one_data

def create_dataset(args, f, pred_shape):
    for key in ["inputs", "outputs", "targets"]:
        f.create_dataset(key, (args.num_sample, args.width, args.width, args.num_ch))
    for key in ["predict_fake", "predict_real"]:
        f.create_dataset(key, (args.num_sample, pred_shape[1], pred_shape[2], pred_shape[3]))

