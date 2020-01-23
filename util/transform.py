import numpy as np
import tensorflow as tf

def upscale(args, element):
    element = element * (np.array(args.max_value) - np.array(args.min_value)) + np.array(args.min_value)
    return element

def list2tensor(args):
    ones = tf.ones([args.width, args.width])
    max_value = tf.stack([args.max_value[ch] * ones for ch in range(args.num_ch)], axis=2)
    max_value = tf.expand_dims(max_value, axis=0)
    min_value = tf.stack([args.min_value[ch] * ones for ch in range(args.num_ch)], axis=2)
    min_value = tf.expand_dims(min_value, axis=0)     
    return max_value, min_value

