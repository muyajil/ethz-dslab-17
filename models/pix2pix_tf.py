''' Code based on https://github.com/yenchenlin/pix2pix-tensorflow/blob/master/model.py
'''

import tensorflow as tf
import numpy as np
from six.moves import xrange


from models.tf_utils import *

from models.abstract_model import ModelConfig
from models.abstract_tensorflow_model import AbstractTensorflowModel

class Pix2pix(AbstractTensorflowModel):
    
    # Use L1 for less blurring
    # 'Regularize' the generator with tasking it to additionally reduce loss to ground truth.
    _l1_lambda = 100
    
    _gen_conv1_filters = 64
    _dis_conv1_filters = 64
    
    _learning_rate = 0.0002
    _momentum = 0.5
    
    
    def train(self, training_set, epochs, validation_set=None):
        """Fits the model parameters to the dataset.

        Args:
            training_set: Instance of Dataset
            epochs: Number of epochs to train
            validation_set: Data on which to evaluate the model.

        """
        with tf.Session() as sess:
            
            # Optimizers
            D_optimizer = tf.train.AdamOptimizer(self._learning_rate, beta1=self._momentum).minimize(self.D_loss, var_list=self.D_vars)
            G_optimizer = tf.train.AdamOptimizer(self._learning_rate, beta1=self._momentum).minimize(self.G_loss, var_list=self.G_vars)
            
            # Initialization
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            
            # Merge summaries
            self.G_summary = tf.summary.merge([self.D_fake_summary, self.G_out_summary, self.D_fake_loss_summary, self.G_loss_summary])
            self.D_summary = tf.summary.merge([self.D_real_summary, self.D_real_loss_summary, self.D_loss_summary])
            self.writer = tf.summary.FileWriter(self._config.log_dir, sess.graph)

            counter = 1
            start_time = time.time()
            
            # TODO: add model restore
            
            for epoch in xrange(epochs):
                for batch in training_set.batch_iter_epoch():
                    
                    # Discriminator
                    _, summary_str = sess.run([D_optimizer, self.D_summary],
                                                    feed_dict={ self.real_input: batch })
                    self.writer.add_summary(summary_str, counter)
                    
                    # Generator
                    _, summary_str = sess.run([G_optimizer, self.D_summary],
                                               feed_dict={ self.real_input: batch })
                    self.writer.add_summary(summary_str, counter)

    def _new_model(self):
        ''' Creates a new pix2pix tensorflow model.
        '''
        # Create Generator and Discriminator
        self.real_input = tf.placeholder(tf.float32, [self._config._batch_size,
                                                 self._config.input_dimensions.height,
                                                 self._config.input_dimensions.width,
                                                 self._config.input_dimensions.depth])
        self.real_output = self.real_input                                    
        self.fake_output = self._generator(self.real_input)
        self.real_AB = tf.concat([self.real_input, self.real_output], 3)
        self.fake_AB = tf.concat([self.real_input, self.fake_output], 3)
        self.D_real, self.D_real_logits = self._discriminator(self.real_AB, reuse=False)
        self.D_fake, self.D_fake_logits = self._discriminator(self.fake_AB, reuse=True)
        
        # Loss functions
        self.D_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_real_logits, labels=tf.ones_like(self.D_real)))
        self.D_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake_logits, labels=tf.zeros_like(self.D_fake)))
        self.D_loss = self.D_real_loss + self.D_fake_loss
        self.G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake_logits, labels=tf.ones_like(self.D_fake))) \
                        + self._l1_lambda * tf.reduce_mean(tf.abs(self.real_output - self.fake_output))
                        
        # Tensorboard
        self.D_real_summary = tf.summary.histogram("d_real", self.D_real)
        self.D_fake_summary = tf.summary.histogram("d_fake", self.D_fake)
        self.G_out_summary = tf.summary.image("g_out", self.fake_output)
        self.D_real_loss_summary = tf.summary.scalar("d_real_loss", self.D_real_loss)
        self.D_fake_loss_summary = tf.summary.scalar("d_fake_loss" , self.D_fake_loss)
        self.D_loss_summary = tf.summary.scalar("d_loss", self.D_loss)
        self.G_loss_summary = tf.summary.scalar("g_loss", self.G_loss)
        
        # Trainable Variables
        train_vars = tf.trainable_variables()
        self.D_vars = [var for var in train_vars if 'd_' in var.name]
        self.G_vars = [var for var in train_vars if 'g_' in var.name]
        
        self.saver = tf.train.Saver()
        return "DUMMY"

        
    def _discriminator(self, image_AB, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False
            dis_dim = _dis_conv1_filters
                                                                                                # EXAMPLE:  if _dis_conv1_filters = 64
                                                                                                # image_AB:     [batch_size, 1024, 1024, 1+1]
            h0 = lrelu(conv2d(image_AB, dis_dim, name='d_h0_conv'))                             # h0:           [batch_size, 512,  512,  64]
            h1 = lrelu(batch_norm(conv2d(h0, dis_dim*2, name='d_h1_conv'), 'd_bn1'))            # h1:           [batch_size, 256,  256,  128]
            h2 = lrelu(batch_norm(conv2d(h1, dis_dim*4, name='d_h2_conv'), 'd_bn2'))            # h2:           [batch_size, 128,  128,  256]
            h3 = lrelu(batch_norm(conv2d(h2, dis_dim*8, stride_height=1, stride_width=1,
                                                        name='d_h3_conv'), 'd_bn3'))            # h3:           [batch_size, 128,  128,  512]
            h4 = linear(tf.reshape(h3, [self._config.batch_size, -1]), 1, 'd_h3_lin')
            return tf.nn.sigmoid(h4), h4
        
    def _generator(self, image):
        '''
        Args:
            image: tensor of shape [batch_size, height, width, depth]
        '''
        with tf.variable_scope("generator") as scope:
            o_c = self._config.input_dimensions.depth
            o_h = self._config.input_dimensions.height
            o_w = self._config.input_dimensions.width   
            h2, h4, h8, h16, h32, h64, h128 = int(o_h/2), int(o_h/4), int(o_h/8), int(o_h/16), int(o_h/32), int(o_h/64), int(o_h/128)
            w2, w4, w8, w16, w32, w64, w128 = int(o_w/2), int(o_w/4), int(o_w/8), int(o_w/16), int(o_w/32), int(o_w/64), int(o_w/128)
            self.gen_dim = self._gen_conv1_filters                                                  
                                                                                                # EXAMPLE:  if _gen_conv1_filters = 64
            # Encoder                                                                           # image:    [batch_size, 1024, 1024, 1]
            e1 = conv2d(image, self.gen_dim, name='g_e1_conv')                                  # e1:       [batch_size, 512, 512, 64]
            e2 = batch_norm(conv2d(lrelu(e1), self.gen_dim*2, name='g_e2_conv'), 'g_bn_e2')     # e2:       [batch_size, 256, 256, 128]
            e3 = batch_norm(conv2d(lrelu(e2), self.gen_dim*4, name='g_e3_conv'), 'g_bn_e3')     # e3:       [batch_size, 128, 128, 256]
            e4 = batch_norm(conv2d(lrelu(e3), self.gen_dim*8, name='g_e4_conv'), 'g_bn_e4')     # e4:       [batch_size, 64,  64,  512]
            e5 = batch_norm(conv2d(lrelu(e4), self.gen_dim*8, name='g_e5_conv'), 'g_bn_e5')     # e5:       [batch_size, 32,  32,  512]
            e6 = batch_norm(conv2d(lrelu(e5), self.gen_dim*8, name='g_e6_conv'), 'g_bn_e6')     # e6:       [batch_size, 16,  16,  512]
            e7 = batch_norm(conv2d(lrelu(e6), self.gen_dim*8, name='g_e7_conv'), 'g_bn_e7')     # e7:       [batch_size, 8,   8,   512]
            e8 = batch_norm(conv2d(lrelu(e7), self.gen_dim*8, name='g_e8_conv'), 'g_bn_e8')     # e8:       [batch_size, 4,   4,   512]
                                                                                                ## TODO:    This can/should? be batch_size, 1,   1,   512]

            # Decoder
            d1 = tf.nn.dropout(batch_norm(deconv2d(tf.nn.relu(e8),
                                                   [None, h128, w128, self.gen_dim*8],
                                                   name='g_d1'), 'g_bn_d1'), 0.5)
            d1 = tf.concat([d1, e7], 3)                                                         # d1:       [batch_size, 8,   8,   512+512]
            d2 = tf.nn.dropout(batch_norm(deconv2d(tf.nn.relu(d1),
                                                   [None, h64, w64, self.gen_dim*8],
                                                   name='g_d2'), 'g_bn_d2'), 0.5)
            d2 = tf.concat([d2, e6], 3)                                                         # d2:       [batch_size, 16,   16,   512+512]
            d3 = tf.nn.dropout(batch_norm(deconv2d(tf.nn.relu(d2),
                                                   [None, h32, w32, self.gen_dim*8],
                                                   name='g_d3'), 'g_bn_d3'), 0.5)
            d3 = tf.concat([d3, e5], 3)                                                         # d3:       [batch_size, 32,   32,   512+512]
            d4 = tf.nn.dropout(batch_norm(deconv2d(tf.nn.relu(d3),
                                                   [None, h16, w16, self.gen_dim*8],
                                                   name='g_d4'), 'g_bn_d4'), 0.5)
            d4 = tf.concat([d4, e4], 3)                                                         # d4:       [batch_size, 64,   64,   512+512]
            d5 = batch_norm(deconv2d(tf.nn.relu(d4),
                                     [None, h8, w8, self.gen_dim*4],
                                     name='g_d5'), 'g_bn_d5')
            d5 = tf.concat([d5, e3], 3)                                                         # d5:       [batch_size, 128,   128,   256+256]
            d6 = batch_norm(deconv2d(tf.nn.relu(d5),
                                     [None, h4, w4, self.gen_dim*2],
                                     name='g_d6'), 'g_bn_d6')
            d6 = tf.concat([d6, e2], 3)                                                         # d6:       [batch_size, 256,   256,   128+128]
            d7 = batch_norm(deconv2d(tf.nn.relu(d6),
                                     [None, h2, w2, self.gen_dim],
                                     name='g_d7'), 'g_bn_d7')
            d7 = tf.concat([d7, e1], 3)                                                         # d7:       [batch_size, 512,   512,   64+64]
            d8 = deconv2d(tf.nn.relu(d7), [None, o_h, o_w, o_c], name='g_d8')                   # d8:       [batch_size, 1024,  1024,  1]
            return tf.nn.tanh(d8)

        
model = Pix2pix()