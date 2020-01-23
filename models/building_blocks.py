import tensorflow as tf
import numpy as np

def conv_layer(inputs, out_ch, stride=2, pad=1):
    with tf.variable_scope("conv"):
        in_ch = inputs.get_shape().as_list()[3]
        filt = tf.get_variable("filter", [4, 4, in_ch, out_ch], dtype=tf.float32)    
        padded_input = tf.pad(inputs, [[0, 0], [pad, pad], [pad, pad], [0, 0]], mode="CONSTANT") 
        # add zero padding in dim 1 and dim 2
        conv = tf.nn.conv2d(padded_input, filt, [1, stride, stride, 1],
        padding="VALID")

    return conv

def deconv_layer(inputs, out_ch, stride=2):
    with tf.variable_scope("deconv"):
        temp_shape = inputs.get_shape().as_list()
        filt = tf.get_variable("filter", [4, 4, out_ch, temp_shape[3]], dtype=tf.float32)
        deconv = tf.nn.conv2d_transpose(inputs, filt, [temp_shape[0], temp_shape[1] * 2,
        temp_shape[2] * 2, out_ch], [1, stride, stride, 1], padding="SAME")

    return deconv

def batchnorm(inputs):
    with tf.variable_scope("batchnorm"):
        in_ch = inputs.get_shape().as_list()[3]
        offset = tf.get_variable("offset", [in_ch], dtype=tf.float32)
        scale = tf.get_variable("scale", [in_ch], dtype=tf.float32)
        mean, variance = tf.nn.moments(inputs, axes=[0, 1, 2], keep_dims=False)
        normalized = tf.nn.batch_normalization(inputs, mean, variance, offset, scale, 
        variance_epsilon=1e-5)
        return normalized

def create_generator(args, inputs):
    # info
    shape = inputs.get_shape().as_list()
    width = shape[1]
    num_ch = shape[3]
    num_layer = int(np.log2(width))
    # all layers
    # inputs -> conv -> act -> (conv -> bn -> act) -> .. 
    # -> (deconv -> bn -> (dropout) -> concate -> act) -> deconv -> tanh

    all_layers = []
    all_layers.append(inputs)
    encoder_id = 0
    decoder_id = num_layer + 1

    # **** Encoder part **** #
    # encoder_1: [batch, width, width, inputs_channels] 
    # => [batch, width / 2, width / 2, ngf]
    with tf.variable_scope("encoder_1"):
        encoder_id += 1
        all_layers.append(conv_layer(inputs, args.ngf, stride=2))
        all_layers.append(tf.nn.leaky_relu(all_layers[-1]))

    # encoder_2: [batch, width / 2, width / 2 , ngf] => [batch, width / 4, width / 4, ngf * 2]        
    # encoder_3: [batch, width / 4, width / 4, ngf * 2] => [batch, width / 8, width / 8, ngf * 4]        
    # encoder_4: [batch, width / 8, width / 8, ngf * 4] => [batch, width / 16, width / 16, ngf * 8] 
    # encoder_i (i > 4): [batch, width / 2**(i-1), width / 2**(i-1), ngf * 8] => [batch, width / 2**i, width / 2**i, ngf * 8] 
    layer_specs = []
    for i in range(num_layer - 1):
        if i < 3:
            layer_specs.append(args.ngf * (2**(i + 1)))
        else:
            layer_specs.append(args.ngf * 8)

    for out_ch in layer_specs:
        encoder_id += 1
        with tf.variable_scope("encoder_%d" % (encoder_id)):
            all_layers.append(conv_layer(all_layers[-1], out_ch, stride=2))
            all_layers.append(batchnorm(all_layers[-1]))       
            all_layers.append(tf.nn.leaky_relu(all_layers[-1]))

    # **** Decoder part **** #

    # decoder_i (i > 4): [batch, width / 2**i, width / 2**i, ngf * 8] => [batch, width / 2**(i-1), width / 2**(i-1), ngf * 8 * 2]                     
    # decoder_4: [batch, width / 16, width / 16, ngf * 8 * 2] => [batch, width / 8, width / 8, ngf * 4 * 2]
    # decoder_3: [batch, width / 8, width / 8, ngf * 4 * 2] => [batch, width / 4, width / 4, ngf * 2 * 2]        
    # decoder_2: [batch, width / 4, width / 4, ngf * 2 * 2] => [batch, width / 2, width / 2, ngf * 1 * 2]        
    layer_specs_num = []
    for i in range(num_layer - 1, 0, -1):
        if i > 3:
            layer_specs_num.append(args.ngf * 8)
        else:
            layer_specs_num.append(args.ngf * (2**(i - 1)))
    # define dropout probs
    layer_specs_d = [0.0 for i in range(len(layer_specs_num))]
    if args.mode == "train":
        layer_specs_d[0] = 0.5
        layer_specs_d[1] = 0.5
    layer_specs = [tuple([layer_specs_num[i], layer_specs_d[i]]) for i in range(len(layer_specs_num))]

    for out_ch, dropout in layer_specs:
        decoder_id -= 1
        skip_id = decoder_id - 1
        with tf.variable_scope("decoder_%d" % (decoder_id)):
            all_layers.append(deconv_layer(all_layers[-1], out_ch))
            all_layers.append(batchnorm(all_layers[-1]))
            if dropout > 0.0:
                all_layers.append(tf.nn.dropout(all_layers[-1], keep_prob=1 - dropout))
  
            all_layers.append(tf.concat([all_layers[-1], all_layers[3 * skip_id - 2]],axis=3))
            all_layers.append(tf.nn.relu(all_layers[-1]))
        
    # decoder_1: [batch, width / 2, width / 2, ngf * 1 * 2] => [batch, width, width, channel]       
    with tf.variable_scope("decoder_1"):
        all_layers.append(deconv_layer(all_layers[-1], num_ch))
        all_layers.append(tf.tanh(all_layers[-1]))

    return all_layers

def create_discriminator(args, inputs, targets):

    # info 
    width = inputs.get_shape().as_list()[1]

    if width == 16:
        num_layer = 1
    elif width == 32:
        num_layer = 2
    else:
        num_layer = 3
    # concate -> conv -> act -> (conv -> bn -> act) -> conv -> sigmoid
    all_layers = []
    all_layers.append(tf.concat([inputs, targets], axis=3))
    # layer_1: [batch, 16, 16, in_channels * 2] => [batch, 8, 8, ndf]
    with tf.variable_scope("layer_1"):
        all_layers.append(conv_layer(all_layers[-1], args.ndf, stride=2))
        all_layers.append(tf.nn.leaky_relu(all_layers[-1]))
    # layer_2: [batch, 8, 8, ndf] => [batch, 7, 7, ndf * 2]    
    for i in range(num_layer):
        with tf.variable_scope("layer_%d" %(i + 2)):
            out_ch = args.ndf * min(2**(i + 1), 8)
            stride = 1 if i == num_layer - 1 else 2 
            all_layers.append(conv_layer(all_layers[-1], out_ch, stride=stride))
            all_layers.append(batchnorm(all_layers[-1]))
            all_layers.append(tf.nn.leaky_relu(all_layers[-1]))
    # layer_3: [batch, 7, 7, ndf * 4] => [batch, 6, 6, 1]
    with tf.variable_scope("layer_%d" %(num_layer + 2)):
        all_layers.append(conv_layer(all_layers[-1], 1, stride=1))
        all_layers.append(tf.nn.sigmoid(all_layers[-1]))
    return all_layers


