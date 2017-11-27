import os
import time
from six.moves import xrange

import numpy as np

from models.tf_utils import *



'''

This model is (trying to be) an implementation of the paper
"Learning to Inpaint for Image Compression" (https://arxiv.org/abs/1709.08855)
by Mohammad Haris Baig, Vladlen Koltun, and Lorenzo Torresani

'''

class Config(object):
    """Configuration for model
    """

    # parameters for logging
    log_dir = None
    
    # parameters for architecture
    batch_size = None
    input_dimensions = None

    # parameters for training
    learning_rate = None
    adam_beta1 = None
    gen_lambda = None
    pretrain_epochs = None


    def __init__(self,
                 batch_size,
                 input_dimensions,
                 log_dir,
                 gen_lambda=100,
                 adam_beta1=0.5,
                 learning_rate=0.0002,
                 pretrain_epochs=1):
                     
        self.batch_size = batch_size
        self.input_dimensions = input_dimensions
        self.log_dir = log_dir
        self.adam_beta1 = adam_beta1
        self.learning_rate = learning_rate
        self.gen_lambda = gen_lambda
        self.pretrain_epochs = pretrain_epochs


class Ops(object):
    
    # architectural ops
    in_img = None

    # losses and metrics
    dis_loss_real = None
    dis_loss_fake = None
    dis_loss = None
    gen_loss_adv = None
    gen_loss_reconstr = None
    gen_loss = None
    psnr = None
    
    # optimizers
    dis_optimizer = None
    gen_optimizer = None
    gen_reconstr_optimizer = None
    
    # initialization
    init_op = None

    # tensorboard
    dis_loss_real_summary = None
    dis_loss_fake_summary = None
    dis_out_real_histo = None
    dis_out_fake_histo = None
    dis_loss_summary = None
    gen_loss_adv_summary = None
    gen_loss_reconstr_summary = None
    gen_loss_summary = None
    dis_summary = None
    gen_summary = None
    gen_reconstr_summary = None
    val_in_out_img_summary = None
    val_psnr_summary = None
    val_summary = None


    
class Res2pix(object):

    _config = None
    _ops = Ops()
    _model_name = None
    
    
    def __init__(self, config=None, restore_path=None):

        self._model_name = str(type(self).__name__)
        config.log_dir = str(os.path.join(config.log_dir, str(int(time.time()))))
        self._config = config
        self._setup_model()
        
        
    def train(self, sess, training_set, epochs, validation_set=None):

        sess.run(self._ops.init_op)
        writer = tf.summary.FileWriter(self._config.log_dir, sess.graph)
        train_step = 1
        start_time = time.time()
        for epoch in xrange(epochs):
            for batch_num, batch in enumerate(training_set.batch_iter(stop_after_epoch=True)):
                
                if epoch < self._config.pretrain_epochs:
                    
                    # generator reconstruction loss
                    _, summary_str = sess.run([self._ops.gen_reconstr_optimizer, self._ops.gen_reconstr_summary], feed_dict={self._ops.in_img: batch})
                    writer.add_summary(summary_str, train_step)
                    
                else:
                    
                    # discriminator
                    _, summary_str = sess.run([self._ops.dis_optimizer, self._ops.dis_summary], feed_dict={self._ops.in_img: batch})
                    writer.add_summary(summary_str, train_step)
                    
                    # generator
                    _, summary_str = sess.run([self._ops.gen_optimizer, self._ops.gen_summary], feed_dict={self._ops.in_img: batch})
                    writer.add_summary(summary_str, train_step)
                    
                print("Epoch: [%2d]\tTrain Step: [%2d]\tBatch: [%2d]\tTime: %4.4f" % (epoch + 1, train_step, batch_num + 1,  time.time() - start_time))
                
                if train_step % 10 == 0:
                    for batch in validation_set.batch_iter(stop_after_epoch=True):
                        summary_str = sess.run(self._ops.val_summary, feed_dict={self._ops.in_img: batch})
                        writer.add_summary(summary_str, global_step=train_step)
                        break # we only do one batch for convenience

                train_step += 1
        writer.close()


    def _setup_model(self):
        
        # input
        self._ops.in_img =  tf.placeholder(tf.float32, [self._config.batch_size,
                                                            self._config.input_dimensions.height,
                                                            self._config.input_dimensions.width,
                                                            self._config.input_dimensions.depth])

        # architecture
        gen_res_preds, gen_residuals = self._generator(self._ops.in_img)
        gen_out = 0
        for res_pred in len(gen_res_preds):
            gen_out += res_pred
            
        dis_out_real, dis_logits_real = self._discriminator(self._ops.in_img, reuse=False)
        dis_out_fake, dis_logits_fake = self._discriminator(gen_out, reuse=True)

        # loss and metrics functions
        self._ops.dis_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_logits_real, labels=tf.ones_like(dis_out_real)))
        self._ops.dis_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_logits_fake, labels=tf.zeros_like(dis_out_fake)))
        self._ops.dis_loss = self._ops.dis_loss_real + self._ops.dis_loss_fake
        self._ops.gen_loss_adv = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_logits_fake, labels=tf.ones_like(dis_out_fake)))       
        
        # Residual encoder loss
        stage_losses = []
        for i in len(gen_residuals):
            stage_losses[i] = tf.reduce_sum(tf.square(gen_residuals[i] - gen_res_preds[i]))
        self._ops.gen_loss_reconstr = tf.reduce_sum(tf.convert_to_tensor(stage_losses))

        self._ops.gen_loss = self._ops.gen_loss_adv + self._config.gen_lambda * self._ops.gen_loss_reconstr
        self._ops.psnr = psnr(self._ops.in_img, gen_out)
        
        # tensorboard
        self._ops.dis_loss_real_summary = tf.summary.scalar("dis_loss_real", self._ops.dis_loss_real)
        self._ops.dis_loss_fake_summary = tf.summary.scalar("dis_loss_fake", self._ops.dis_loss_fake)
        self._ops.dis_out_real_histo = tf.summary.histogram("dis_out_real", dis_out_real)
        self._ops.dis_out_fake_histo = tf.summary.histogram("dis_out_fake", dis_out_fake)
        self._ops.dis_loss_summary = tf.summary.scalar("dis_loss", self._ops.dis_loss)
        self._ops.gen_loss_adv_summary = tf.summary.scalar("gen_loss_adv", self._ops.gen_loss_adv)
        self._ops.gen_loss_reconstr_summary = tf.summary.scalar("gen_loss_reconstr", self._ops.gen_loss_reconstr)
        self._ops.gen_loss_summary = tf.summary.scalar("gen_loss", self._ops.gen_loss)
        self._ops.val_psnr_summary = tf.summary.scalar("val_psnr", self._ops.psnr)
        self._ops.val_in_out_img_summary = tf.summary.image("val_in_out_img", tf.concat([self._ops.in_img, gen_out], 1))
        self._ops.dis_summary = tf.summary.merge([self._ops.dis_loss_real_summary,
                                                  self._ops.dis_out_real_histo,
                                                  self._ops.dis_loss_summary])
        self._ops.gen_summary = tf.summary.merge([self._ops.dis_loss_fake_summary,
                                                  self._ops.dis_out_fake_histo,
                                                  self._ops.gen_loss_adv_summary,
                                                  self._ops.gen_loss_reconstr_summary,
                                                  self._ops.gen_loss_summary])
        self._ops.gen_reconstr_summary = tf.summary.merge([self._ops.gen_loss_reconstr_summary])
        self._ops.val_summary = tf.summary.merge([self._ops.val_in_out_img_summary,
                                                  self._ops.val_psnr_summary])

        # trainable variables and optimizers
        train_vars = tf.trainable_variables()
        dis_vars = [var for var in train_vars if 'd_' in var.name]
        gen_vars = [var for var in train_vars if 'g_' in var.name]
        self._ops.dis_optimizer = tf.train.AdamOptimizer(self._config.learning_rate, beta1=self._config.adam_beta1).minimize(self._ops.dis_loss, var_list=dis_vars)
        self._ops.gen_optimizer = tf.train.AdamOptimizer(self._config.learning_rate, beta1=self._config.adam_beta1).minimize(self._ops.gen_loss, var_list=gen_vars)
        self._ops.gen_reconstr_optimizer = tf.train.AdamOptimizer(self._config.learning_rate, beta1=self._config.adam_beta1).minimize(self._ops.gen_loss_reconstr, var_list=gen_vars)
        
        # initialization
        self._ops.init_op = tf.global_variables_initializer()


    def _discriminator(self, image, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert not tf.get_variable_scope().reuse

            h0 = lrelu(conv2d(image, 64, name='d_h0_conv'))
            h1 = lrelu(batch_norm(conv2d(h0, 64 * 2, name='d_h1_conv'), name='d_bn1'))
            h2 = lrelu(batch_norm(conv2d(h1, 64 * 3, name='d_h2_conv'), name='d_bn2'))
            h3 = lrelu(batch_norm(conv2d(h2, 64 * 8, stride_height=1, stride_width=1, name='d_h3_conv'), name='d_bn3'))
            h4 = linear(tf.reshape(h3, [self._config.batch_size, -1]), 1, scope='d_h3_lin')
            return tf.nn.sigmoid(h4), h4
            
            
    def _residual_encoder(self, image):
        stages = 6
        with tf.variable_scope("generator") as scope:
            
            res[0] = image
            stage_preds = []
            for s in range(stages)[1:]:
                stage_preds[s-1] = _residual_encoder_stage(res[s-1])
                res[s] = res[s-1] - stage_preds[s-1]
            return stage_preds, res
            
            
    def _residual_encoder_stage(self, res_in):
        e1 = tf.nn.relu(batch_norm(conv2d(res_in, 64, kernel_height=3, kernel_width=3, stride_height=1, stride_width=1, stddev=0.02, name='g_e1_conv'), name='g_bn_e1'))
        e2 = tf.nn.relu(batch_norm(conv2d(e1, 128, kernel_height=3, kernel_width=3, stride_height=2, stride_width=2, stddev=0.02, name='g_e2_conv'), name='g_bn_e2'))
        e3 = tf.nn.relu(batch_norm(conv2d(e2, 128, kernel_height=3, kernel_width=3, stride_height=1, stride_width=1, stddev=0.02, name='g_e3_conv'), name='g_bn_e3'))
        e4 = tf.nn.relu(batch_norm(conv2d(e3, 256, kernel_height=3, kernel_width=3, stride_height=2, stride_width=2, stddev=0.02, name='g_e4_conv'), name='g_bn_e4'))
        e5 = tf.nn.relu(batch_norm(conv2d(e4, 256, kernel_height=3, kernel_width=3, stride_height=1, stride_width=1, stddev=0.02, name='g_e5_conv'), name='g_bn_e5'))
        e6 = tf.nn.relu(batch_norm(conv2d(e5, 256, kernel_height=3, kernel_width=3, stride_height=2, stride_width=2, stddev=0.02, name='g_e6_conv'), name='g_bn_e6'))
        e7 = tf.nn.tanh(conv2d(e6, 8, kernel_height=1, kernel_width=1, stride_height=2, stride_width=2, stddev=0.02, name='g_e6_conv'))


        res_out = None
        return res_out
            
    def _generator_pix2pix(self, image):
        with tf.variable_scope("generator") as scope:
            o_c = self._config.input_dimensions.depth
            o_h = self._config.input_dimensions.height
            o_w = self._config.input_dimensions.width
            h2, h4, h8, h16, h32, h64, h128 = \
                int(o_h / 2), int(o_h / 4), int(o_h / 8), int(o_h / 16), int(o_h / 32), int(o_h / 64), int(o_h / 128)
            w2, w4, w8, w16, w32, w64, w128 = \
                int(o_w / 2), int(o_w / 4), int(o_w / 8), int(o_w / 16), int(o_w / 32), int(o_w / 64), int(o_w / 128)

            # encoder
            e1 = conv2d(image, 64, name='g_e1_conv')
            e2 = batch_norm(conv2d(lrelu(e1), 64 * 2, name='g_e2_conv'), name='g_bn_e2')
            e3 = batch_norm(conv2d(lrelu(e2), 64 * 4, name='g_e3_conv'), name='g_bn_e3')
            e4 = batch_norm(conv2d(lrelu(e3), 64 * 8, name='g_e4_conv'), name='g_bn_e4')
            e5 = batch_norm(conv2d(lrelu(e4), 64 * 8, name='g_e5_conv'), name='g_bn_e5')
            e6 = batch_norm(conv2d(lrelu(e5), 64 * 8, name='g_e6_conv'), name='g_bn_e6')
            e7 = batch_norm(conv2d(lrelu(e6), 64 * 8, name='g_e7_conv'), name='g_bn_e7')
            e8 = batch_norm(conv2d(lrelu(e7), 64 * 8, name='g_e8_conv'), name='g_bn_e8')

            # decoder
            d1 = tf.concat([tf.nn.dropout(batch_norm(deconv2d(tf.nn.relu(e8), [self._config.batch_size, h128, w128, 64 * 8], name='g_d1'), name='g_bn_d1'), 0.5), e7], 3)
            d2 = tf.concat([tf.nn.dropout(batch_norm(deconv2d(tf.nn.relu(d1), [self._config.batch_size, h64,  w64,  64 * 8], name='g_d2'), name='g_bn_d2'), 0.5), e6], 3)
            d3 = tf.concat([tf.nn.dropout(batch_norm(deconv2d(tf.nn.relu(d2), [self._config.batch_size, h32,  w32,  64 * 8], name='g_d3'), name='g_bn_d3'), 0.5), e5], 3)
            d4 = tf.concat([tf.nn.dropout(batch_norm(deconv2d(tf.nn.relu(d3), [self._config.batch_size, h16,  w16,  64 * 8], name='g_d4'), name='g_bn_d4'), 0.5), e4], 3)
            d5 = tf.concat([tf.nn.dropout(batch_norm(deconv2d(tf.nn.relu(d4), [self._config.batch_size, h8,   w8,   64 * 4], name='g_d5'), name='g_bn_d5'), 0.5), e3], 3)
            d6 = tf.concat([tf.nn.dropout(batch_norm(deconv2d(tf.nn.relu(d5), [self._config.batch_size, h4,   w4,   64 * 2], name='g_d6'), name='g_bn_d6'), 0.5), e2], 3)
            d7 = tf.concat([batch_norm(deconv2d(tf.nn.relu(d6), [self._config.batch_size, h2, w2, 64], name='g_d7'), name='g_bn_d7'), e1], 3)
            d8 = deconv2d(tf.nn.relu(d7), [self._config.batch_size, o_h, o_w, o_c], name='g_d8')
            return tf.nn.tanh(d8)


