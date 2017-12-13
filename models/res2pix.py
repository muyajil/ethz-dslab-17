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
    steps_between_val = None

    # parameters for loging
    log_dir = None
    
    # parameters for architecture
    batch_size = None
    patch_size = None
    input_dimensions = None
    stages = None
    conv_filter_size = None

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
                 adam_beta1=0.9,
                 learning_rate=0.001,
                 pretrain_epochs=1,
                 stages=1,
                 debug=False,
                 show_jpeg=True,
                 steps_between_val=100,
                 patch_size=128,
                 conv_filter_size=3):
                     
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
        self.steps_between_val = steps_between_val
        self.patch_size = patch_size
        self.conv_filter_size = conv_filter_size


class Ops(object):
    
    # architectural ops
    in_img = None
    binary_representations = []
    gen_preds = None
    gen_patch_preds = None
    patches = None

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
    gen_loss_reconstr_stages_summaries = None
    img_summary = None


    
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
        
        # evaluate jpeg performance
        if self._config.show_jpeg:
            print("Evaluation jpeg performance..")
            for bpp, psnr, mssim in self._evaluate_jpeg(sess, validation_set):
                if bpp > 0.5:
                    continue
                psnr_summary = tf.Summary()
                mssim_summary = tf.Summary()
                psnr_summary.value.add(tag='jpeg_psnr', simple_value=psnr)
                mssim_summary.value.add(tag='jpeg_mssim', simple_value=mssim)
                writer.add_summary(psnr_summary, global_step=int(bpp*10000))
                writer.add_summary(mssim_summary, global_step=int(bpp*10000))
            writer.flush()
              
        # training  
        print("Starting with training..")
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
                
                # evaluation
                if train_step % self._config.steps_between_val == 0:
                    
                    # show images
                    for batch in validation_set.batch_iter(stop_after_epoch=True):
                        summary_str = sess.run(self._ops.img_summary, feed_dict={self._ops.in_img: batch})
                        writer.add_summary(summary_str, global_step=train_step)
                        break # we only do one batch
                    
                    # psnr and mssim
                    psnr_summaries, mssim_summaries = self._evaluation(sess, validation_set)
                    for i in range(len(psnr_summaries)):
                        writer.add_summary(psnr_summaries[i], global_step=train_step)
                        writer.add_summary(mssim_summaries[i], global_step=train_step)

                train_step += 1
      
        writer.close()
        
        
    def _evaluate_jpeg(self, sess, validation_set):
        with open('jpeg_evaluation.txt', 'w') as file:
            qualities = []
            for quality in range(0, 101, 1)[1:]:
                print("measuring performance for quality = " + str(quality))
                avg_psnrs = []
                avg_mssims = []
                avg_bpps = []
                for batch in validation_set.batch_iter(stop_after_epoch=True):
                    originals = sess.run(self._ops.patches, feed_dict={self._ops.in_img: batch})
                    
                    original_size_pixel = originals.shape[1] * originals.shape[2]
                    
                    psnrs = []
                    mssims = []
                    bpps = []
                    npatches = int(self._config.input_dimensions.width / self._config.patch_size)
                    for i in range(self._config.batch_size * npatches):
                        
                        # get jpeg reconstruction
                        pil_original = Image.fromarray(np.squeeze(((originals[i]+1)/2)*255).astype('uint8'), 'L')  # we may loose precision here
                        pil_original.save("out.jpg", "JPEG", quality=quality, optimize=True, progressive=True)   
                        pil_jpeg = Image.open("out.jpg")
                        jpeg = np.array(pil_jpeg).astype(originals[i].dtype)
                        jpeg = ((jpeg / 255) * 2) - 1
                        file_size_bits = os.path.getsize("out.jpg")*8
                        
                        # measure performance
                        psnrs.append(compare_psnr(np.squeeze(originals[i]), jpeg, data_range=2))
                        mssims.append(compare_ssim(np.squeeze(originals[i]), jpeg, data_range=2, win_size=9))
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
                file.write(str(bpp) + ' ' + str(psnr) + ' ' + str(mssim) + '\n')
                qualities.append((bpp, psnr, mssim))
                
        print("length of qualities list = " + str(len(qualities)))
        return sorted(qualities, key=lambda x: x[0])
                    

    def _evaluation(self, sess, validation_set):
        
        print("Evaluating...")
        avg_psnrs_stages = [[] for _ in range(self._config.stages)]
        avg_mssims_stages = [[] for _ in range(self._config.stages)]

        for batch in validation_set.batch_iter(stop_after_epoch=True):
            
            # get images from tf session  
            originals, reconstructions = sess.run([self._ops.patches, self._ops.gen_patch_preds], feed_dict={self._ops.in_img: batch})
    
            for j in range(self._config.stages):
                psnrs_stage = []
                mssims_stage = []
                npatches = int(self._config.input_dimensions.width / self._config.patch_size)
                for i in range(self._config.batch_size * npatches):
                    psnrs_stage.append(compare_psnr(np.squeeze(originals[i]), np.squeeze(reconstructions[j][i]), data_range=2))
                    mssims_stage.append(compare_ssim(np.squeeze(originals[i]), np.squeeze(reconstructions[j][i]), data_range=2, win_size=9))
                avg_psnrs_stages[j].append(sum(psnrs_stage)/len(psnrs_stage))
                avg_mssims_stages[j].append(sum(mssims_stage)/len(mssims_stage))

        psnr_summaries = []
        mssim_summaries = []
        for j in range(self._config.stages):
            psnr_summary = tf.Summary()
            mssim_summary = tf.Summary()
            psnr_summary.value.add(tag='avg_val_psnr_stage' + str(j), simple_value=sum(avg_psnrs_stages[j])/len(avg_psnrs_stages[j]))
            mssim_summary.value.add(tag='avg_val_mssim_stage' + str(j), simple_value=sum(avg_mssims_stages[j])/len(avg_mssims_stages[j]))
            psnr_summaries.append(psnr_summary)
            mssim_summaries.append(mssim_summary)

        return psnr_summaries, mssim_summaries


    def _setup_model(self):
        
        npatches = int(self._config.input_dimensions.width / self._config.patch_size)
        
        # input
        with tf.variable_scope("input"):
            self._ops.in_img = tf.placeholder(tf.float32, [self._config.batch_size,
                                                            self._config.input_dimensions.height,
                                                            self._config.input_dimensions.width,
                                                            self._config.input_dimensions.depth])
            print("number of patches = " + str(npatches))
            print("shape of input image = " + str(self._ops.in_img.get_shape()))
            self._ops.patches = tf.concat(tf.split(self._ops.in_img, npatches, 2), 0)
                                                            
        with tf.variable_scope("generator"):
            self._ops.gen_patch_preds = self._generator_R2I_decode(self._ops.patches)
            self._ops.gen_preds = [tf.concat(tf.split(stage_patch_pred, npatches, 0), 2) for stage_patch_pred in self._ops.gen_patch_preds]

        with tf.variable_scope("discriminator"):
            dis_out_real, dis_logits_real = self._discriminator(self._ops.patches, reuse=False)
            dis_out_fake, dis_logits_fake = self._discriminator(self._ops.gen_patch_preds[-1], reuse=True)


        losses = []
        with tf.variable_scope("optimization"):
            # loss and metrics functions
            with tf.variable_scope("GAN_loss"):
                self._ops.dis_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_logits_real, labels=tf.ones_like(dis_out_real)))
                self._ops.dis_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_logits_fake, labels=tf.zeros_like(dis_out_fake)))
                self._ops.dis_loss = self._ops.dis_loss_real + self._ops.dis_loss_fake
                self._ops.gen_loss_adv = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_logits_fake, labels=tf.ones_like(dis_out_fake)))       
            
            with tf.variable_scope("reconstruction_loss"):
                loss = 0
                patch_residuals = []
                i = 0
                for pred in self._ops.gen_patch_preds:
                    stage_residual = self._ops.patches - pred
                    # stage_loss = tf.reduce_mean(tf.reduce_sum(tf.abs(stage_residual), [1, 2, 3]))
                    stage_loss = tf.reduce_mean(tf.reduce_sum(tf.square(stage_residual), [1, 2, 3]))
                    losses.append(stage_loss)
                    patch_residuals.append(stage_residual)
                    loss = loss + stage_loss
                    i += 1
                self._ops.gen_loss_reconstr = loss

            self._ops.gen_loss = self._ops.gen_loss_adv + self._config.gen_lambda * self._ops.gen_loss_reconstr

        self._ops.gen_loss_reconstr_stages_summaries = []
        for i in range(len(losses)):
            self._ops.gen_loss_reconstr_stages_summaries.append(tf.summary.scalar("gen_loss_reconstr_stage" + str(i), losses[i]))

        
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
                                      
        reconstr_sum = list(self._ops.gen_loss_reconstr_stages_summaries)
        reconstr_sum.append(self._ops.gen_loss_reconstr_summary)
        self._ops.gen_reconstr_summary = tf.summary.merge(reconstr_sum)
        
        # image summary
        with tf.variable_scope("image_summary"):
            b, w, h, d = self._ops.binary_representations[0].get_shape().as_list()
            bitmaps = [tf.concat(tf.split(tf.concat(tf.split(stage_rep, npatches, 0), 2), int(d), 3), 1) for stage_rep in self._ops.binary_representations]
            bitmaps.append(tf.zeros_like(bitmaps[0]))
            images = list(self._ops.gen_preds)
            images.append(self._ops.in_img)
            residuals = [tf.concat(tf.split(patch_residual, npatches, 0), 2) for patch_residual in patch_residuals]
            residuals.append(tf.zeros_like(self._ops.in_img))
            images_c = tf.concat(images, 1)
            bitmaps_c = tf.concat(bitmaps, 1)
            residuals_c = tf.concat(residuals, 1)
            image_summary =  tf.concat([images_c, bitmaps_c,  residuals_c], 2)
        
        self._ops.img_summary = tf.summary.image("val_img_summary", image_summary, max_outputs=16)

        # trainable variables and optimizers
        train_vars = tf.trainable_variables()
        dis_vars = [var for var in train_vars if 'd_' in var.name]
        gen_vars = [var for var in train_vars if 'g_' in var.name]

        with tf.variable_scope("optimization"):
            with tf.variable_scope("GAN_optimizer_discriminator"):
                dis_opti = tf.train.AdamOptimizer(self._config.learning_rate, beta1=self._config.adam_beta1)
                dis_grads_and_vars = dis_opti.compute_gradients(self._ops.dis_loss, var_list=dis_vars)
                self._ops.dis_optimizer = dis_opti.apply_gradients(dis_grads_and_vars)
            
            with tf.variable_scope("GAN_optimizer_generator"):
                gen_opti = tf.train.AdamOptimizer(self._config.learning_rate, beta1=self._config.adam_beta1)
                gen_grads_and_vars = gen_opti.compute_gradients(self._ops.gen_loss, var_list=gen_vars)
                self._ops.gen_optimizer = gen_opti.apply_gradients(gen_grads_and_vars)
            
            with tf.variable_scope("reconstruction_optimizer_generator"):
                gen_reconstr_opti = tf.train.AdamOptimizer(self._config.learning_rate, beta1=self._config.adam_beta1)
                gen_reconstr_grads_and_vars = gen_reconstr_opti.compute_gradients(self._ops.gen_loss_reconstr, var_list=gen_vars)
                self._ops.gen_reconstr_optimizer = gen_reconstr_opti.apply_gradients(gen_reconstr_grads_and_vars)
            
        # initialization
        self._ops.init_op = tf.global_variables_initializer()
        
        # debug
        if self._config.debug:
            print("Shape of input placeholder = " + str(self._ops.in_img.get_shape()))
            print("Shape of discriminator (fake) output = " + str(dis_out_fake.get_shape()))


    def _discriminator(self, image, reuse=False):
        with tf.variable_scope("discriminator_instance") as scope:
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert not tf.get_variable_scope().reuse
                
            batchsize, height, width, channels = image.get_shape().as_list()


            h0 = lrelu(conv2d(image, 64, name='d_h0_conv'))
            h1 = lrelu(batch_norm(conv2d(h0, 64 * 2, name='d_h1_conv'), name='d_bn1'))
            h2 = lrelu(batch_norm(conv2d(h1, 64 * 3, name='d_h2_conv'), name='d_bn2'))
            h3 = lrelu(batch_norm(conv2d(h2, 64 * 8, stride_height=1, stride_width=1, name='d_h3_conv'), name='d_bn3'))
            h4 = linear(tf.reshape(h3, [batchsize, -1]), 1, scope='d_h3_lin')
            return tf.nn.sigmoid(h4), h4
            
            
    def _generator_R2I_decode(self, image):
        stage_preds = []
        current_prediction = 0
        current_conv_links = [0, 0, 0]
        current_residual = image
        for s in range(self._config.stages + 1)[1:]:
            with tf.variable_scope("stage_" + str(s)):
                current_prediction, current_conv_links = self._R2I_decode_stage(current_residual, current_conv_links)
                current_residual = image - current_prediction
                stage_preds.append(current_prediction)
        
        # compute bpp
        bin_dim = 1
        for dim in self._ops.binary_representations[0].get_shape().as_list()[1:]:
            bin_dim *= dim
        self._code_bits = (bin_dim * self._config.stages)
            
        return stage_preds
            
    def _R2I_decode_stage(self, res_in, prev_convs):

        batchsize, height, width, channels = res_in.get_shape().as_list()
        c_height = int(height / 8)
        c_width = int(width / 8)
        
        # encoder
        e1 = tf.nn.relu(batch_norm(conv2d(res_in, 64, kernel_height=self._config.conv_filter_size, kernel_width=self._config.conv_filter_size, stride_height=1, stride_width=1, stddev=0.02, name='g_e1_conv'), name='g_bn_e1'), name='g_ridge_e1')
        e2 = tf.nn.relu(batch_norm(conv2d(e1, 128, kernel_height=self._config.conv_filter_size, kernel_width=self._config.conv_filter_size, stride_height=2, stride_width=2, stddev=0.02, name='g_e2_conv'), name='g_bn_e2'), name='g_ridge_e2')
        e3 = tf.nn.relu(batch_norm(conv2d(e2, 128, kernel_height=self._config.conv_filter_size, kernel_width=self._config.conv_filter_size, stride_height=1, stride_width=1, stddev=0.02, name='g_e3_conv'), name='g_bn_e3'), name='g_ridge_e3')
        e4 = tf.nn.relu(batch_norm(conv2d(e3, 256, kernel_height=self._config.conv_filter_size, kernel_width=self._config.conv_filter_size, stride_height=2, stride_width=2, stddev=0.02, name='g_e4_conv'), name='g_bn_e4'), name='g_ridge_e4')
        e5 = tf.nn.relu(batch_norm(conv2d(e4, 256, kernel_height=self._config.conv_filter_size, kernel_width=self._config.conv_filter_size, stride_height=1, stride_width=1, stddev=0.02, name='g_e5_conv'), name='g_bn_e5'), name='g_ridge_e5')
        e6 = tf.nn.relu(batch_norm(conv2d(e5, 256, kernel_height=self._config.conv_filter_size, kernel_width=self._config.conv_filter_size, stride_height=2, stride_width=2, stddev=0.02, name='g_e6_conv'), name='g_bn_e6'), name='g_ridge_e6')
        e7 = tf.nn.tanh(conv2d(e6, 8, kernel_height=1, kernel_width=1, stride_height=1, stride_width=1, stddev=0.02, name='g_e7_conv'), name='g_ridge_e7')
        
        # binarization
        self._ops.binary_representations.append(binarization(e7))

        # decoder
        d1 = conv2d(self._ops.binary_representations[-1], 256, kernel_height=self._config.conv_filter_size, kernel_width=self._config.conv_filter_size, stride_height=1, stride_width=1, stddev=0.02, name='g_d1_conv')
        d2 = tf.nn.relu(batch_norm(conv2d(d1, 256, kernel_height=self._config.conv_filter_size, kernel_width=self._config.conv_filter_size, stride_height=1, stride_width=1, stddev=0.02, name='g_d2_conv'), name='g_bn_d2'), name='g_ridge_d2')
        
        d3 = deconv2d(d2, [batchsize, c_height*2, c_width*2, 256], kernel_height=2, kernel_width=2, stride_height=2, stride_width=2, stddev=0.02, name="g_d3_deconv")
        d3_prev = 0
        if prev_convs[0] != 0:
            d3_prev = tf.nn.relu(batch_norm(conv2d(prev_convs[0], 256, kernel_height=self._config.conv_filter_size, kernel_width=self._config.conv_filter_size, stride_height=1, stride_width=1, stddev=0.02, name='g_d3_conv_prev'), name='g_bn_d3_prev'), name='g_ridge_d3_prev')
        d3_comb = tf.nn.tanh(d3 + d3_prev, name='g_ridge_d3') 
        
        d4 = tf.nn.relu(batch_norm(conv2d(d3_comb, 128, kernel_height=self._config.conv_filter_size, kernel_width=self._config.conv_filter_size, stride_height=1, stride_width=1, stddev=0.02, name='g_d4_conv'), name='g_bn_d4'), name='g_ridge_d4')
        
        d5 = deconv2d(d4, [batchsize, c_height*4, c_width*4, 128], kernel_height=2, kernel_width=2, stride_height=2, stride_width=2, stddev=0.02, name="g_d5_deconv")
        d5_prev = 0
        if prev_convs[1] != 0:
            d5_prev = tf.nn.relu(batch_norm(conv2d(prev_convs[1], 128, kernel_height=self._config.conv_filter_size, kernel_width=self._config.conv_filter_size, stride_height=1, stride_width=1, stddev=0.02, name='g_d5_conv_prev'), name='g_bn_d5_prev'), name='g_ridge_d5_prev')
        d5_comb = tf.nn.tanh(d5 + d5_prev, name='g_ridge_d5') 
        
        d6 = tf.nn.relu(batch_norm(conv2d(d5_comb, 64, kernel_height=self._config.conv_filter_size, kernel_width=self._config.conv_filter_size, stride_height=1, stride_width=1, stddev=0.02, name='g_d6_conv'), name='g_bn_d6'), name='g_ridge_d6')

        d7 = deconv2d(d6, [batchsize, c_height*8, c_width*8, 64], kernel_height=2, kernel_width=2, stride_height=2, stride_width=2, stddev=0.02, name="g_d7_deconv")
        d7_prev = 0
        if prev_convs[2] != 0:
            d7_prev = tf.nn.relu(batch_norm(conv2d(prev_convs[2], 64, kernel_height=self._config.conv_filter_size, kernel_width=self._config.conv_filter_size, stride_height=1, stride_width=1, stddev=0.02, name='g_d7_conv_prev'), name='g_bn_d7_prev'), name='g_ridge_d7_prev')
        d7_comb = tf.nn.tanh(d7 + d7_prev, name='g_ridge_d7')  
        
        pred = tf.nn.tanh(conv2d(d7_comb, channels, kernel_height=self._config.conv_filter_size, kernel_width=self._config.conv_filter_size, stride_height=1, stride_width=1, stddev=0.02, name='g_d8_conv'), name='g_ridge_d8')

        convs = [d3, d5, d7]

        return pred, convs