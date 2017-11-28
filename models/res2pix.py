import os
import time
from six.moves import xrange

import numpy as np

from models.tf_utils import *

from skimage.measure import compare_psnr, compare_ssim
from PIL import Image



'''

This model is (trying to be) an implementation of the paper
"Learning to Inpaint for Image Compression" (https://arxiv.org/abs/1709.08855)
by Mohammad Haris Baig, Vladlen Koltun, and Lorenzo Torresani

'''

class Config(object):
    """Configuration for model
    """
    
    debug = None
    show_jpeg = None

    # parameters for loging
    log_dir = None
    
    # parameters for architecture
    batch_size = None
    input_dimensions = None
    stages = None

    # parameters for training
    learning_rate = None
    adam_beta1 = None
    gen_lambda = None
    pretrain_epochs = None


    def __init__(self,
                 batch_size,
                 input_dimensions,
                 log_dir,
                 gen_lambda=1,
                 adam_beta1=0.5,
                 learning_rate=0.0002,
                 pretrain_epochs=1,
                 stages=1,
                 debug=False,
                 show_jpeg=True):
                     
        self.batch_size = batch_size
        self.input_dimensions = input_dimensions
        self.log_dir = log_dir
        self.adam_beta1 = adam_beta1
        self.learning_rate = learning_rate
        self.gen_lambda = gen_lambda
        self.pretrain_epochs = pretrain_epochs
        self.stages = stages
        self.debug = debug
        self.show_jpeg = show_jpeg


class Ops(object):
    
    # architectural ops
    in_img = None
    binary_representations = []

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
    _code_bits = None
    
    
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
        
        # evaluate jpeg performance
        if self._config.show_jpeg:
            print("Evaluation jpeg performance..")
            for bpp, psnr, mssim in self._evaluate_jpeg(sess, validation_set):
                psnr_summary = tf.Summary()
                mssim_summary = tf.Summary()
                psnr_summary.value.add(tag='jpeg_psnr', simple_value=psnr)
                mssim_summary.value.add(tag='jpeg_mssim', simple_value=mssim)
                writer.add_summary(psnr_summary, global_step=bpp)
                writer.add_summary(mssim_summary, global_step=bpp)
              
        # training  
        print("Starting with training..")
        bpp = float(self._code_bits) / (self._config.input_dimensions.height * self._config.input_dimensions.width)
        print("Bits per Pixel = " + str(bpp))
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
                
                # evaluation
                if train_step % 10 == 0:
                    
                    # show images
                    for batch in validation_set.batch_iter(stop_after_epoch=True):
                        summary_str = sess.run(self._ops.val_in_out_img_summary, feed_dict={self._ops.in_img: batch})
                        writer.add_summary(summary_str, global_step=train_step)
                        break # we only do one batch
                    
                    # psnr and mssim
                    psnr_summary, mssim_summary = self._evaluation(sess, validation_set)
                    writer.add_summary(psnr_summary, global_step=train_step)
                    writer.add_summary(mssim_summary, global_step=train_step)

                train_step += 1
      
        writer.close()
        
        
    def _evaluate_jpeg(self, sess, validation_set):
        qualities = []
        for quality in range(0, 101, 10)[1:]:
            print("measuring performance for quality = " + str(quality))
            avg_psnrs = []
            avg_mssims = []
            avg_bpps = []
            for batch in validation_set.batch_iter(stop_after_epoch=True):
                originals = sess.run(self._ops.in_img, feed_dict={self._ops.in_img: batch})
                
                original_size_pixel = originals.shape[1] * originals.shape[2]
                
                psnrs = []
                mssims = []
                bpps = []
                for i in range(self._config.batch_size):
                    
                    # get jpeg reconstruction
                    
                    pil_original = Image.fromarray(np.squeeze(((originals[i]+1)/2)*255).astype('uint8'), 'L')  # we may loose precision here
                    pil_original.save("out.jpg", "JPEG", quality=quality, optimize=True, progressive=True)   
                    pil_jpeg = Image.open("out.jpg")
                    jpeg = np.array(pil_jpeg).astype(originals[i].dtype)
                    jpeg = ((jpeg / 255) * 2) - 1
                    file_size_bits = os.path.getsize("out.jpg")*8
                    
                    # measure performance
                    psnrs.append(compare_psnr(np.squeeze(originals[i]), jpeg, data_range=2))
                    mssims.append(compare_ssim(np.squeeze(originals[i]), jpeg, data_range=2))
                    bpps.append(float(file_size_bits) / original_size_pixel)
                    os.remove("out.jpg")
                    
                avg_psnr = sum(psnrs)/len(psnrs)
                avg_psnrs.append(avg_psnr)
                avg_mssim = sum(mssims)/len(mssims)
                avg_mssims.append(avg_mssim)
                avg_bpp = sum(bpps)/len(bpps)
                avg_bpps.append(avg_bpp)
            psnr = sum(avg_psnrs)/len(avg_psnrs)
            mssim = sum(avg_mssims)/len(avg_mssims)
            bpp = sum(avg_bpps)/len(avg_bpps)
            print("bpp = " + str(bpp))
            print("psnr = " + str(psnr))
            print("mssim = " + str(mssim))

            qualities.append((bpp, psnr, mssim))
            
        print("length of qualities list = " + str(len(qualities)))
        return sorted(qualities, key=lambda x: x[0])
                    

    def _evaluation(self, sess, validation_set):
        
        print("Evaluating...")
        avg_psnrs = []
        avg_mssims = []
        for batch in validation_set.batch_iter(stop_after_epoch=True):
            originals, reconstructions = sess.run([self._ops.in_img, self._ops.gen_out], feed_dict={self._ops.in_img: batch})
            
            if self._config.debug:
                print("Shape of fetched images = " + str(originals.shape))
            
            psnrs = []
            mssims = []
            for i in range(self._config.batch_size):
                psnrs.append(compare_psnr(np.squeeze(originals[i]), np.squeeze(reconstructions[i]), data_range=2))
                mssims.append(compare_ssim(np.squeeze(originals[i]), np.squeeze(reconstructions[i]), data_range=2))
            avg_psnr = sum(psnrs)/len(psnrs)
            avg_psnrs.append(avg_psnr)
            avg_mssim = sum(mssims)/len(mssims)
            avg_mssims.append(avg_mssim)
        psnr = sum(avg_psnrs)/len(avg_psnrs)
        mssim = sum(avg_mssims)/len(avg_mssims)
        
        psnr_summary = tf.Summary()
        mssim_summary = tf.Summary()
        psnr_summary.value.add(tag='avg_val_psnr', simple_value=psnr)
        mssim_summary.value.add(tag='avg_val_mssim', simple_value=mssim)

        return psnr_summary, mssim_summary


    def _setup_model(self):
        
        # input
        self._ops.in_img =  tf.placeholder(tf.float32, [self._config.batch_size,
                                                            self._config.input_dimensions.height,
                                                            self._config.input_dimensions.width,
                                                            self._config.input_dimensions.depth])
                                                            
        # architecture
        gen_res_preds, gen_residuals = self._generator(self._ops.in_img)
        self._ops.gen_out = 0
        for res_pred in gen_res_preds:
            self._ops.gen_out += res_pred
            
        dis_out_real, dis_logits_real = self._discriminator(self._ops.in_img, reuse=False)
        dis_out_fake, dis_logits_fake = self._discriminator(self._ops.gen_out, reuse=True)

        # loss and metrics functions
        self._ops.dis_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_logits_real, labels=tf.ones_like(dis_out_real)))
        self._ops.dis_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_logits_fake, labels=tf.zeros_like(dis_out_fake)))
        self._ops.dis_loss = self._ops.dis_loss_real + self._ops.dis_loss_fake
        self._ops.gen_loss_adv = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_logits_fake, labels=tf.ones_like(dis_out_fake)))       
        
        # Residual encoder loss
        stage_losses = []
        for res in gen_residuals[1:]:
            stage_losses.append(tf.reduce_mean(tf.reduce_sum(tf.square(res), [1, 2, 3])))
        self._ops.gen_loss_reconstr = tf.reduce_sum(tf.convert_to_tensor(stage_losses))

        self._ops.gen_loss = self._ops.gen_loss_adv + self._config.gen_lambda * self._ops.gen_loss_reconstr
        self._ops.psnr = psnr(self._ops.in_img, self._ops.gen_out)
        
        # tensorboard
        self._ops.dis_loss_real_summary = tf.summary.scalar("dis_loss_real", self._ops.dis_loss_real)
        self._ops.dis_loss_fake_summary = tf.summary.scalar("dis_loss_fake", self._ops.dis_loss_fake)
        self._ops.dis_out_real_histo = tf.summary.histogram("dis_out_real", dis_out_real)
        self._ops.dis_out_fake_histo = tf.summary.histogram("dis_out_fake", dis_out_fake)
        self._ops.dis_loss_summary = tf.summary.scalar("dis_loss", self._ops.dis_loss)
        self._ops.gen_loss_adv_summary = tf.summary.scalar("gen_loss_adv", self._ops.gen_loss_adv)
        self._ops.gen_loss_reconstr_summary = tf.summary.scalar("gen_loss_reconstr", self._ops.gen_loss_reconstr)
        self._ops.gen_loss_summary = tf.summary.scalar("gen_loss", self._ops.gen_loss)
        self._ops.dis_summary = tf.summary.merge([self._ops.dis_loss_real_summary,
                                                  self._ops.dis_out_real_histo,
                                                  self._ops.dis_loss_summary])
        self._ops.gen_summary = tf.summary.merge([self._ops.dis_loss_fake_summary,
                                                  self._ops.dis_out_fake_histo,
                                                  self._ops.gen_loss_adv_summary,
                                                  self._ops.gen_loss_reconstr_summary,
                                                  self._ops.gen_loss_summary])
        self._ops.gen_reconstr_summary = tf.summary.merge([self._ops.gen_loss_reconstr_summary])
        self._ops.val_psnr_summary = tf.summary.scalar("val_psnr", self._ops.psnr)
        self._ops.val_bitcode_histo = tf.summary.histogram("val_bitcode_histogram", self._ops.binary_representations)
        
        
        
        self._ops.val_in_out_img_summary = tf.summary.image("val_in_out_img", tf.concat([self._ops.in_img, self._ops.gen_out], 1))
        self._ops.val_summary = tf.summary.merge([self._ops.val_in_out_img_summary,
                                                  self._ops.val_psnr_summary,
                                                  self._ops.val_bitcode_histo])

        # trainable variables and optimizers
        train_vars = tf.trainable_variables()
        dis_vars = [var for var in train_vars if 'd_' in var.name]
        gen_vars = [var for var in train_vars if 'g_' in var.name]

        dis_opti = tf.train.AdamOptimizer(self._config.learning_rate, beta1=self._config.adam_beta1)
        dis_grads_and_vars = dis_opti.compute_gradients(self._ops.dis_loss, var_list=dis_vars)
        self._ops.dis_optimizer = dis_opti.apply_gradients(dis_grads_and_vars)
        
        gen_opti = tf.train.AdamOptimizer(self._config.learning_rate, beta1=self._config.adam_beta1)
        gen_grads_and_vars = gen_opti.compute_gradients(self._ops.gen_loss, var_list=gen_vars)
        self._ops.gen_optimizer = gen_opti.apply_gradients(gen_grads_and_vars)
        
        gen_reconstr_opti = tf.train.AdamOptimizer(self._config.learning_rate, beta1=self._config.adam_beta1)
        gen_reconstr_grads_and_vars = gen_reconstr_opti.compute_gradients(self._ops.gen_loss_reconstr, var_list=gen_vars)
        self._ops.gen_reconstr_optimizer = gen_reconstr_opti.apply_gradients(gen_reconstr_grads_and_vars)
        
        # initialization
        self._ops.init_op = tf.global_variables_initializer()
        
        # debug
        if self._config.debug:
            print("Shape of input placeholder = " + str(self._ops.in_img.get_shape()))
            print("Number of residuals (including input image) = " + str(len(gen_residuals)))
            print("Shape of first residual prediction = " + str(gen_res_preds[0].get_shape()))
            print("Shape of first residual = " + str(gen_residuals[0].get_shape()))
            print("Shape of generator output = " + str(self._ops.gen_out.get_shape()))
            print("Shape of discriminator (fake) output = " + str(dis_out_fake.get_shape()))


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
            
            
    def _generator(self, image):
        with tf.variable_scope("generator") as scope:
            
            stage_preds = []
            res = []
            res.append(image)
            for s in range(self._config.stages + 1)[1:]:
                stage_preds.append(self._residual_encoder_stage(res[s-1], name="stage_" + str(s)))
                res.append(res[s-1] - stage_preds[s-1])
                
            # compute bpp
            bin_dim = 1
            for dim in self._ops.binary_representations[0].get_shape().as_list()[1:]:
                bin_dim *= dim
            self._code_bits = (bin_dim * self._config.stages)
                
            return stage_preds, res
            
            
    def _residual_encoder_stage(self, res_in, name="stage"):
        with tf.variable_scope(name):

            batchsize, height, width, channels = res_in.get_shape().as_list()
            c_height = int(height / 8)
            c_width = int(width / 8)
            
            # encoder
            e1 = tf.nn.relu(batch_norm(conv2d(res_in, 64, kernel_height=3, kernel_width=3, stride_height=1, stride_width=1, stddev=0.02, name='g_e1_conv'), name='g_bn_e1'))
            e2 = tf.nn.relu(batch_norm(conv2d(e1, 128, kernel_height=3, kernel_width=3, stride_height=2, stride_width=2, stddev=0.02, name='g_e2_conv'), name='g_bn_e2'))
            e3 = tf.nn.relu(batch_norm(conv2d(e2, 128, kernel_height=3, kernel_width=3, stride_height=1, stride_width=1, stddev=0.02, name='g_e3_conv'), name='g_bn_e3'))
            e4 = tf.nn.relu(batch_norm(conv2d(e3, 256, kernel_height=3, kernel_width=3, stride_height=2, stride_width=2, stddev=0.02, name='g_e4_conv'), name='g_bn_e4'))
            e5 = tf.nn.relu(batch_norm(conv2d(e4, 256, kernel_height=3, kernel_width=3, stride_height=1, stride_width=1, stddev=0.02, name='g_e5_conv'), name='g_bn_e5'))
            e6 = tf.nn.relu(batch_norm(conv2d(e5, 256, kernel_height=3, kernel_width=3, stride_height=2, stride_width=2, stddev=0.02, name='g_e6_conv'), name='g_bn_e6'))
            e7 = tf.nn.tanh(conv2d(e6, 8, kernel_height=1, kernel_width=1, stride_height=1, stride_width=1, stddev=0.02, name='g_e7_conv'))
            
            # binarization
            self._ops.binary_representations.append(binarization(e7))

            # decoder
            d1 = conv2d(self._ops.binary_representations[-1], 256, kernel_height=3, kernel_width=3, stride_height=1, stride_width=1, stddev=0.02, name='g_d1_conv')
            d2 = tf.nn.relu(batch_norm(conv2d(d1, 256, kernel_height=3, kernel_width=3, stride_height=1, stride_width=1, stddev=0.02, name='g_d2_conv'), name='g_bn_d2'))
            d3 = deconv2d(d2, [self._config.batch_size, c_height*2, c_width*2, 256], kernel_height=2, kernel_width=2, stride_height=2, stride_width=2, stddev=0.02, name="g_d3_deconv")
            d4 = tf.nn.relu(batch_norm(conv2d(d3, 128, kernel_height=3, kernel_width=3, stride_height=1, stride_width=1, stddev=0.02, name='g_d4_conv'), name='g_bn_d4'))
            d5 = deconv2d(d4, [self._config.batch_size, c_height*4, c_width*4, 128], kernel_height=2, kernel_width=2, stride_height=2, stride_width=2, stddev=0.02, name="g_d5_deconv")
            d6 = tf.nn.relu(batch_norm(conv2d(d5, 64, kernel_height=3, kernel_width=3, stride_height=1, stride_width=1, stddev=0.02, name='g_d6_conv'), name='g_bn_d6'))
            d7 = deconv2d(d6, [self._config.batch_size, c_height*8, c_width*8, 64], kernel_height=2, kernel_width=2, stride_height=2, stride_width=2, stddev=0.02, name="g_d7_deconv")
            d8 = tf.nn.tanh(conv2d(d7, channels, kernel_height=3, kernel_width=3, stride_height=1, stride_width=1, stddev=0.02, name='g_d8_conv'))
    
            # debug
            if self._config.debug:
                print("Stage: Shape of input = " + str(res_in.get_shape()))
                print("Stage: Shape of e1 = " + str(e1.get_shape()))
                print("Stage: Shape of e2 = " + str(e2.get_shape()))
                print("Stage: Shape of e3 = " + str(e3.get_shape()))
                print("Stage: Shape of e4 = " + str(e4.get_shape()))
                print("Stage: Shape of e5 = " + str(e5.get_shape()))
                print("Stage: Shape of e6 = " + str(e6.get_shape()))
                print("Stage: Shape of e7 = " + str(e7.get_shape()))
                print("Stage: Shape of binary representation = " + str(self._ops.binary_representations[-1].get_shape()))
                print("Stage: Shape of d1 = " + str(d1.get_shape()))
                print("Stage: Shape of d2 = " + str(d2.get_shape()))
                print("Stage: Shape of d3 = " + str(d3.get_shape()))
                print("Stage: Shape of d4 = " + str(d4.get_shape()))
                print("Stage: Shape of d5 = " + str(d5.get_shape()))
                print("Stage: Shape of d6 = " + str(d6.get_shape()))
                print("Stage: Shape of d7 = " + str(d7.get_shape()))
                print("Stage: Shape of d8 = " + str(d8.get_shape()))
    
            return d8
            
            
'''
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
'''
