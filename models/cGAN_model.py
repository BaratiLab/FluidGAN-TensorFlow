import tensorflow as tf
from .building_blocks import create_generator, create_discriminator

class CGANModel(object):

    def __init__(self, args, inputs, targets):
        self.args = args
        self.inputs = inputs
        self.targets = targets
        # create model
        self.create_model()
        self.outputs = self.all_g_layers[-1]
        self.predict_real = self.all_d_real_layers[-1]
        self.predict_real_map = self.all_d_real_layers[-2]
        self.pred_shape = self.predict_real_map.get_shape().as_list()
        self.predict_fake = self.all_d_fake_layers[-1]
        self.predict_fake_map = self.all_d_fake_layers[-2]
        if args.mode == "train":
            # compute loss
            self.compute_loss()
            # train options
            self.train()
            # define a train option -> update loss, increase global step and train gen(discri)
            self.train_op=tf.group(self.update_losses, self.incr_global_step, self.gen_train)
        
    def create_model(self):
        with tf.variable_scope("generator"):
            self.all_g_layers = create_generator(self.args, self.inputs)
        with tf.name_scope("real_discriminator"):
            with tf.variable_scope("discriminator"):
                self.all_d_real_layers = create_discriminator(self.args, 
                self.inputs, self.targets)
        with tf.name_scope("fake_discriminator"):
            with tf.variable_scope("discriminator", reuse=True):
                self.all_d_fake_layers = create_discriminator(self.args, 
                self.inputs, self.all_g_layers[-1])    
    
    def compute_loss(self):
        with tf.name_scope("discriminator_loss"):
            self.discrim_loss = tf.reduce_mean(-(tf.log(self.predict_real + self.args.eps) + 
            tf.log(1 - self.predict_fake + self.args.eps))) # will return a scalar
        with tf.name_scope("generator_loss"):
            self.gen_loss_GAN = tf.reduce_mean(-tf.log(self.predict_fake + self.args.eps))
            self.gen_loss_L1 = tf.reduce_mean(tf.abs(self.targets - self.outputs))
            self.gen_loss = self.gen_loss_GAN * self.args.gan_weight + self.gen_loss_L1 * self.args.l1_weight   
    
    # **** Gradients, Loss and Optimizer **** #
    # Since we want to save the (gradient, variable) pair, we explicitly use compute_grad and apply_grad
    def train(self):
        with tf.name_scope("discriminator_train"):
            self.discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
            self.discrim_optim = tf.train.AdamOptimizer(self.args.disc_lr, self.args.disc_beta1)
            self.discrim_grads_and_vars = self.discrim_optim.compute_gradients(self.discrim_loss, var_list=self.discrim_tvars)
            # discrim_train is an opt
            self.discrim_train = self.discrim_optim.apply_gradients(self.discrim_grads_and_vars)    
        with tf.name_scope("generator_train"):
            with tf.control_dependencies([self.discrim_train]):
                self.gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]   
                self.gen_optim = tf.train.AdamOptimizer(self.args.gen_lr, self.args.gen_beta1)
                self.gen_grads_and_vars = self.gen_optim.compute_gradients(self.gen_loss, var_list=self.gen_tvars)
                self.gen_train = self.gen_optim.apply_gradients(self.gen_grads_and_vars)
        # apply ema to losses
        self.ema = tf.train.ExponentialMovingAverage(decay=0.99)
        self.update_losses = self.ema.apply([self.discrim_loss, self.gen_loss_GAN, self.gen_loss_L1])    
        self.global_step = tf.train.get_or_create_global_step()
        self.incr_global_step = tf.assign(self.global_step, self.global_step+1)    
        


        
            

