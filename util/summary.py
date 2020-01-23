import tensorflow as tf

def add_summaries(args, model):
    # loss
    tf.summary.scalar("discriminator_loss", model.discrim_loss)
    tf.summary.scalar("generator_loss_GAN", model.gen_loss_GAN)
    tf.summary.scalar("generator_loss_L1", model.gen_loss_L1)
    # outputs
    with tf.name_scope("display_images"):
        for ch in range(args.num_ch):
            tf.summary.image("output_channel%d" % (ch), tf.image.convert_image_dtype(model.outputs[:, :, :, ch: ch + 1],
                             dtype=tf.uint8))
            tf.summary.image("intput_channel%d" % (ch), tf.image.convert_image_dtype(model.inputs[:, :, :, ch: ch + 1],
                             dtype=tf.uint8))
            tf.summary.image("target_channel%d" % (ch), tf.image.convert_image_dtype(model.targets[:, :, :, ch: ch + 1],
                             dtype=tf.uint8))            
    # prediction of discriminator
    with tf.name_scope("predict_real_summary"):
        tf.summary.image("predict_real", tf.image.convert_image_dtype(model.predict_real, dtype=tf.uint8))
    with tf.name_scope("predict_fake_summary"):
        tf.summary.image("predict_fake", tf.image.convert_image_dtype(model.predict_fake, dtype=tf.uint8))
    sum_ops = tf.summary.merge_all()
    return sum_ops