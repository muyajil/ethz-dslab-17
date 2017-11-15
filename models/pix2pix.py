import os
from six.moves import xrange
import time
from models.tf_utils import *
import numpy as np
from scipy.misc import imsave, toimage


class Config(object):
    """Configuration for model
    """

    batch_size = None
    input_dimensions = None
    log_dir = None

    link_flags = None
    gen_train_times = None
    dis_filter_multipliers = None
    gen_filter_multipliers = None

    l1_lambda = None
    smooth = None
    gen_conv1_filters = None
    dis_conv1_filters = None
    learning_rate = None
    momentum = None
    gen_grad_max = 100000  # TODO: find suitbale value for this
    gen_grad_min = -gen_grad_max

    def __init__(self,
                 batch_size,
                 input_dimensions,
                 log_dir,
                 l1_lambda=100,
                 smooth = 0.0,
                 gen_conv1_filters=64,
                 dis_conv1_filters=64,
                 learning_rate=0.0002,
                 momentum=0.5,
                 link_flags=[True for _ in range(7)],
                 gen_train_times=1,
                 dis_filter_multipliers=[2,4,8],
                 gen_filter_multipliers=[2,4,8,8,8,8,8]):
        self.batch_size = batch_size
        self.input_dimensions = input_dimensions
        self.log_dir = log_dir
        self.smooth = smooth
        self.l1_lambda = l1_lambda
        self.gen_conv1_filters = gen_conv1_filters
        self.dis_conv1_filters = dis_conv1_filters
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.link_flags = link_flags
        self.gen_train_times = gen_train_times
        self.dis_filter_multipliers = dis_filter_multipliers
        self.gen_filter_multipliers = gen_filter_multipliers

        # Compute Compression Rate
        h = input_dimensions.height
        w = input_dimensions.width
        gf = gen_conv1_filters
        compressed_size = (h/256)*(w/256)*gf*gen_filter_multipliers[6]
        if link_flags[0]:
            compressed_size += (h/128)*(w/128)*gf*gen_filter_multipliers[5]
        if link_flags[1]:
            compressed_size += (h/64)*(w/64)*gf*gen_filter_multipliers[4]
        if link_flags[2]:
            compressed_size += (h/32)*(w/32)*gf*gen_filter_multipliers[3]
        if link_flags[3]:
            compressed_size += (h/16)*(w/16)*gf*gen_filter_multipliers[2]
        if link_flags[4]:
            compressed_size += (h/8)*(w/8)*gf*gen_filter_multipliers[1]
        if link_flags[5]:
            compressed_size += (h/4)*(w/4)*gf*gen_filter_multipliers[0]
        if link_flags[6]:
            compressed_size += (h/2)*(w/2)*gf*1
        compression_rate = (h*w) / compressed_size
        print("Compressed size = " + str(compressed_size))
        print("Compression Rate = " + str(compression_rate))

# TODO: Add object for summaries


class Pix2PixOps(object):
    input_image = None

    # Losses
    dis_loss = None
    dis_fake_loss = None
    dis_real_loss = None
    gen_loss = None
    gen_l1_loss = None

    # Other metrics
    gen_psnr = None

    # Variables
    dis_vars = None
    gen_vars = None

    # Tensorboard summaries
    dis_loss_summary = None
    dis_real_loss_summary = None
    dis_fake_loss_summary = None
    gen_loss_summary = None
    gen_psnr_summary = None
    concatenated_images = None
    dis_real_pred_histo = None
    dis_fake_pred_histo = None
    gen_l1_loss_summary = None

    gen_filter_images_summary = None

class Pix2pix(object):
    # Use L1 for less blurring
    # 'Regularize' the generator with tasking it to additionally reduce loss to ground truth.

    _ops = Pix2PixOps()
    _saver = None
    _model_name = None
    _config = None
    _restore = False

    def __init__(self, config=None, restore_path=None):
        """
        Args:
            config: Hyperparameters of the model
            restore_path: Path to a stored model state.
                     If None, a new model will be created.
        """
        self._model_name = str(type(self).__name__)
        if restore_path is not None:
            config.log_dir = restore_path
            self._restore = True
        else:
            config.log_dir = str(os.path.join(config.log_dir, str(int(time.time()))))
        self._config = config
        self._setup_model()

    def save(self, sess, step):
        self._saver.save(sess, os.path.join(self._config.log_dir, "model.ckpt"), global_step=step)

    def restore(self, sess):
        # TODO: Restore global step from checkpoint name

        print("Reading checkpoint...")
        ckpt = tf.train.get_checkpoint_state(self._config.log_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self._saver.restore(sess, os.path.join(self._config.log_dir, ckpt_name))
            return True
        else:
            return False

    def train(self, training_set, epochs, validation_set=None):
        """Fits the model parameters to the dataset.

        Args:
            training_set: Instance of Dataset
            epochs: Number of epochs to train
            validation_set: Data on which to evaluate the model.

        """

        with tf.Session() as sess:

            # Optimizer for Discriminator
            dis_optimizer = tf.train.AdamOptimizer(self._config.learning_rate, beta1=self._config.momentum).minimize(
                self._ops.dis_loss, var_list=self._ops.dis_vars)

            # Optimizer for Generator
            gen_optimizer = tf.train.AdamOptimizer(self._config.learning_rate, beta1=self._config.momentum).minimize(
                self._ops.gen_loss, var_list=self._ops.gen_vars)

            # Optimizer for Generator before adding discriminator
            gen_l1_optimizer = tf.train.AdamOptimizer(self._config.learning_rate, beta1=self._config.momentum).minimize(
                 self._ops.gen_l1_loss, var_list=self._ops.gen_vars)


            # Initialization
            init_op = tf.global_variables_initializer()
            sess.run(init_op)

            dis_summary = tf.summary.merge(
                [self._ops.dis_real_loss_summary, self._ops.dis_real_pred_histo,
                 self._ops.dis_loss_summary])

            gen_summary = tf.summary.merge(
                [self._ops.dis_fake_pred_histo, self._ops.dis_fake_loss_summary,
                 self._ops.gen_loss_summary, self._ops.gen_l1_loss_summary,
                 self._ops.gen_filter_images_summary])

            writer = tf.summary.FileWriter(self._config.log_dir, sess.graph)

            train_step = 1
            start_time = time.time()

            if not self._restore:
                print("Will create new model...")
            else:
                if self.restore(sess):
                    print("Load succeeded...")
                else:
                    print("Load failed...")

            window = 100
            pretrain_epochs = 1
            gen_l1_losses = list(100 for _ in range(window))
            gen_l1_losses_index = 0
            pre_train = True
            pre_train_thresh = 0.05

            for epoch in xrange(epochs):
                for batch_num, batch in enumerate(training_set.batch_iter(stop_after_epoch=True)):

                    if epoch < pretrain_epochs:
                
                        # Train only the generator on the l1 loss
                        _, l1_summary, l1_loss = sess.run(
                            [gen_l1_optimizer, self._ops.gen_l1_loss_summary, self._ops.gen_l1_loss],
                            feed_dict={self._ops.input_image: batch})
                        writer.add_summary(l1_summary, train_step)

                        gen_l1_losses[gen_l1_losses_index] = l1_loss
                        gen_l1_losses_index += 1
                        gen_l1_losses_index %= window

                    else:
                        if True:
                            # Discriminator
                            _, summary_str = sess.run([dis_optimizer, dis_summary],
                                                  feed_dict={self._ops.input_image: batch})
                            writer.add_summary(summary_str, train_step)

                            for _ in range(self._config.gen_train_times):
                                # Generator
                                _, summary_str, l1_loss = sess.run([gen_optimizer, gen_summary,  self._ops.gen_l1_loss],
                                                      feed_dict={self._ops.input_image: batch})
                                writer.add_summary(summary_str, train_step)
                                gen_l1_losses[gen_l1_losses_index] = l1_loss
                                gen_l1_losses_index += 1
                                gen_l1_losses_index %= window

                        else:
                            # Generator
                            _, summary_str, l1_loss = sess.run([gen_optimizer, gen_summary, self._ops.gen_l1_loss],
                                                               feed_dict={self._ops.input_image: batch})
                            writer.add_summary(summary_str, train_step)
                            gen_l1_losses[gen_l1_losses_index] = l1_loss
                            gen_l1_losses_index += 1
                            gen_l1_losses_index %= window

                    print(
                        "Epoch: [%2d]\tTrain Step: [%2d]\tBatch: [%2d]\tTime: %4.4f" % (
                            epoch + 1,
                            train_step,
                            batch_num + 1,
                            time.time() - start_time))

                    # if train_step % 500 == 0:
                        # DISABLED FOR EXPERIMENTS: self.save(sess, train_step)
                    if train_step % 100 == 0:
                        images_summary, loss_summary, psnr_summary = self.validate(sess, validation_set, train_step)
                        writer.add_summary(images_summary, global_step=train_step)
                        writer.add_summary(loss_summary, global_step=train_step)
                        writer.add_summary(psnr_summary, global_step=train_step)
                    train_step = train_step + 1
                self.save(sess, train_step)
            writer.close()

    def validate(self, sess, validation_set, train_step):
        print("Validating...")
        images_summary = tf.Summary()
        loss_summary = tf.Summary()
        psnr_summary = tf.Summary()
        validation_losses = []
        validation_pnsrs = []
        batch_images = []
        for batch in validation_set.batch_iter(stop_after_epoch=True):
            batch_images.append(sess.run([self._ops.concatenated_images], feed_dict={self._ops.input_image: batch})[0])
            validation_losses.append(self._ops.gen_loss.eval({self._ops.input_image: batch}))
            validation_pnsrs.append(self._ops.gen_psnr.eval({self._ops.input_image: batch}))
        avg_val_loss = sum(validation_losses) / len(validation_losses)
        avg_val_psnr = sum(validation_pnsrs) / len(validation_pnsrs)
        single_images = []
        for image in batch_images:
            single_images.extend(np.split(image, self._config.batch_size))

        if not os.path.exists('{}/images'.format(self._config.log_dir)):
            os.makedirs('{}/images'.format(self._config.log_dir))

        for image_id, image in enumerate(single_images):
            toimage(np.squeeze(image), cmin=-1, cmax=1).save(
                '{}/images/validation_image_{}_{}.png'.format(self._config.log_dir, image_id, train_step))
            encoded_image = open(
                '{}/images/validation_image_{}_{}.png'.format(self._config.log_dir, image_id, train_step), 'rb').read()
            images_summary.value.add(tag='validation_images/' + str(image_id),
                                     image=tf.Summary.Image(encoded_image_string=encoded_image))

        loss_summary.value.add(tag='avg_validation_loss', simple_value=avg_val_loss)
        psnr_summary.value.add(tag='avg_val_gen_psnr', simple_value=avg_val_psnr)

        return images_summary, loss_summary, psnr_summary

    def _setup_model(self):
        """ Creates a new pix2pix tensorflow model.
        """
        # Create Generator and Discriminator
        self._ops.input_image = tf.placeholder(tf.float32, [self._config.batch_size,
                                                            self._config.input_dimensions.height,
                                                            self._config.input_dimensions.width,
                                                            self._config.input_dimensions.depth])

        generator_output = self._generator(self._ops.input_image, link_flags=self._config.link_flags)

        real_images = tf.concat([self._ops.input_image, self._ops.input_image], 3)
        fake_images = tf.concat([self._ops.input_image, generator_output], 3)
        print("fake_images shape: " + str(fake_images.get_shape()))
        dis_real_pred, dis_real_logits = self._discriminator(real_images, reuse=False)
        dis_fake_pred, dis_fake_logits = self._discriminator(fake_images, reuse=True)

        print("dis_real_logits = " + str(dis_real_logits))
        print("dis_fake_logits = " + str(dis_fake_logits))

        # Loss functions
        self._ops.dis_real_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_real_logits, labels=tf.ones_like(dis_real_pred)*(1-self._config.smooth)))
        self._ops.dis_fake_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_fake_logits, labels=tf.zeros_like(dis_fake_pred)))
        self._ops.dis_loss = self._ops.dis_fake_loss + self._ops.dis_real_loss
        self._ops.gen_l1_loss = tf.reduce_mean(tf.abs(self._ops.input_image - generator_output))
        self._ops.gen_loss = \
            tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_fake_logits, labels=tf.ones_like(dis_fake_pred))) \
            + self._config.l1_lambda * self._ops.gen_l1_loss
        # Metrics
        self._ops.gen_psnr = psnr(self._ops.input_image, generator_output)

        # Tensorboard
        self._ops.gen_psnr_summary = tf.summary.scalar("gen_psnr", self._ops.gen_psnr)
        self._ops.dis_loss_summary = tf.summary.scalar("dis_loss", self._ops.dis_loss)
        self._ops.gen_loss_summary = tf.summary.scalar("gen_loss", self._ops.gen_loss)
        self._ops.dis_real_loss_summary = tf.summary.scalar("dis_real_loss", self._ops.dis_real_loss)
        self._ops.dis_fake_loss_summary = tf.summary.scalar("dis_fake_loss", self._ops.dis_fake_loss)
        self._ops.dis_real_pred_histo = tf.summary.histogram("dis_real_pred", dis_real_pred)
        self._ops.dis_fake_pred_histo = tf.summary.histogram("dis_fake_pred", dis_fake_pred)
        self._ops.gen_l1_loss_summary = tf.summary.scalar("gen_l1_loss", self._ops.gen_l1_loss)
        self._ops.concatenated_images = tf.concat([generator_output, self._ops.input_image], 1)
        with tf.variable_scope("", reuse=True):
            kernel1 = tf.get_variable("generator/g_e1_conv/w")
            kernel2 = tf.get_variable("generator/g_e2_conv/w")
            kernel3 = tf.get_variable("generator/g_e3_conv/w")
            filter1 = tf.reshape(kernel1, [-1, 5, 5, 1])[0]
            filter2 = tf.reshape(kernel2, [-1, 5, 5, 1])[0]
            filter3 = tf.reshape(kernel3, [-1, 5, 5, 1])[0]
            filters = tf.stack([filter1, filter2, filter3])
            self._ops.gen_filter_images_summary = tf.summary.image("gen_filters", filters)

        # Trainable Variables
        train_vars = tf.trainable_variables()
        self._ops.dis_vars = [var for var in train_vars if 'd_' in var.name]
        self._ops.gen_vars = [var for var in train_vars if 'g_' in var.name]

        self._saver = tf.train.Saver()

    def _discriminator(self, images, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert not tf.get_variable_scope().reuse

            """
            EXAMPLE:  if _dis_conv1_filters = 64
            image_AB:     [batch_size, 1024, 1024, 1+1]
            h0:           [batch_size, 512,  512,  64]
            h1:           [batch_size, 256,  256,  128]
            h2:           [batch_size, 128,  128,  256]
            h3:           [batch_size, 128,  128,  512]
            """

            h0 = lrelu(conv2d(images, self._config.dis_conv1_filters, name='d_h0_conv'))
            h1 = lrelu(batch_norm(conv2d(h0, self._config.dis_conv1_filters * self._config.dis_filter_multipliers[0], name='d_h1_conv'), name='d_bn1'))
            h2 = lrelu(batch_norm(conv2d(h1, self._config.dis_conv1_filters * self._config.dis_filter_multipliers[1], name='d_h2_conv'), name='d_bn2'))
            h3 = lrelu(batch_norm(conv2d(h2, self._config.dis_conv1_filters * self._config.dis_filter_multipliers[2], stride_height=1, stride_width=1,
                                         name='d_h3_conv'), name='d_bn3'))
            h4 = linear(tf.reshape(h3, [self._config.batch_size, -1]), 1, scope='d_h3_lin')
            return tf.nn.sigmoid(h4), h4

    def _generator(self, image, link_flags=[True for _ in range(7)]):
        """
        Args:
            image: tensor of shape [batch_size, height, width, depth]
        """
        with tf.variable_scope("generator") as scope:
            o_c = self._config.input_dimensions.depth
            o_h = self._config.input_dimensions.height
            o_w = self._config.input_dimensions.width
            h2, h4, h8, h16, h32, h64, h128 = \
                int(o_h / 2), int(o_h / 4), int(o_h / 8), int(o_h / 16), int(o_h / 32), int(o_h / 64), int(o_h / 128)
            w2, w4, w8, w16, w32, w64, w128 = \
                int(o_w / 2), int(o_w / 4), int(o_w / 8), int(o_w / 16), int(o_w / 32), int(o_w / 64), int(o_w / 128)
            self.gen_dim = self._config.gen_conv1_filters

            # Encoder
            e1 = conv2d(image, self.gen_dim, name='g_e1_conv')
            e2 = batch_norm(conv2d(lrelu(e1), self.gen_dim * self._config.gen_filter_multipliers[0], name='g_e2_conv'), name='g_bn_e2')
            e3 = batch_norm(conv2d(lrelu(e2), self.gen_dim * self._config.gen_filter_multipliers[1], name='g_e3_conv'), name='g_bn_e3')
            e4 = batch_norm(conv2d(lrelu(e3), self.gen_dim * self._config.gen_filter_multipliers[2], name='g_e4_conv'), name='g_bn_e4')
            e5 = batch_norm(conv2d(lrelu(e4), self.gen_dim * self._config.gen_filter_multipliers[3], name='g_e5_conv'), name='g_bn_e5')
            e6 = batch_norm(conv2d(lrelu(e5), self.gen_dim * self._config.gen_filter_multipliers[4], name='g_e6_conv'), name='g_bn_e6')
            e7 = batch_norm(conv2d(lrelu(e6), self.gen_dim * self._config.gen_filter_multipliers[5], name='g_e7_conv'), name='g_bn_e7')
            e8 = batch_norm(conv2d(lrelu(e7), self.gen_dim * self._config.gen_filter_multipliers[6], name='g_e8_conv'), name='g_bn_e8')

            print("e9 dimension: " + str(e8.get_shape()))

            # Decoder
            d1 = tf.nn.dropout(batch_norm(deconv2d(tf.nn.relu(e8),
                                                   [self._config.batch_size, h128, w128, self.gen_dim * self._config.gen_filter_multipliers[5]],
                                                   name='g_d1'), name='g_bn_d1'), 0.5)
            if link_flags[0]:                                         
                d1 = tf.concat([d1, e7], 3)
            d2 = tf.nn.dropout(batch_norm(deconv2d(tf.nn.relu(d1),
                                                   [self._config.batch_size, h64, w64, self.gen_dim * self._config.gen_filter_multipliers[4]],
                                                   name='g_d2'), name='g_bn_d2'), 0.5)
            if link_flags[1]:                                         
                d2 = tf.concat([d2, e6], 3)
            d3 = tf.nn.dropout(batch_norm(deconv2d(tf.nn.relu(d2),
                                                   [self._config.batch_size, h32, w32, self.gen_dim * self._config.gen_filter_multipliers[3]],
                                                   name='g_d3'), name='g_bn_d3'), 0.5)
            if link_flags[2]:                                        
                d3 = tf.concat([d3, e5], 3)
            d4 = tf.nn.dropout(batch_norm(deconv2d(tf.nn.relu(d3),
                                                   [self._config.batch_size, h16, w16, self.gen_dim * self._config.gen_filter_multipliers[2]],
                                                   name='g_d4'), name='g_bn_d4'), 0.5)
            if link_flags[3]: 
                d4 = tf.concat([d4, e4], 3)
            d5 = batch_norm(deconv2d(tf.nn.relu(d4),
                                     [self._config.batch_size, h8, w8, self.gen_dim * self._config.gen_filter_multipliers[1]],
                                     name='g_d5'), name='g_bn_d5')
            if link_flags[4]: 
                d5 = tf.concat([d5, e3], 3)
            d6 = batch_norm(deconv2d(tf.nn.relu(d5),
                                     [self._config.batch_size, h4, w4, self.gen_dim * self._config.gen_filter_multipliers[0]],
                                     name='g_d6'), name='g_bn_d6')
            if link_flags[5]: 
                d6 = tf.concat([d6, e2], 3)
            d7 = batch_norm(deconv2d(tf.nn.relu(d6),
                                     [self._config.batch_size, h2, w2, self.gen_dim],
                                     name='g_d7'), name='g_bn_d7')
            if link_flags[6]: 
                d7 = tf.concat([d7, e1], 3)
            d8 = deconv2d(tf.nn.relu(d7), [self._config.batch_size, o_h, o_w, o_c], name='g_d8')
            return tf.nn.tanh(d8)
