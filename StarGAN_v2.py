from ops import *
from utils import *
import time
from tensorflow.contrib.data import prefetch_to_device, shuffle_and_repeat, map_and_batch # tf 1.13

import numpy as np
from glob import glob
from tqdm import tqdm

class StarGAN_v2() :
    def __init__(self, sess, args):
        self.model_name = 'StarGAN_v2'
        self.sess = sess
        self.phase = args.phase
        self.checkpoint_dir = args.checkpoint_dir
        self.sample_dir = args.sample_dir
        self.result_dir = args.result_dir
        self.log_dir = args.log_dir
        self.dataset_name = args.dataset
        self.dataset_path = os.path.join('./dataset', self.dataset_name)
        self.augment_flag = args.augment_flag

        self.decay_flag = args.decay_flag
        self.decay_iter = args.decay_iter

        self.gpu_num = args.gpu_num
        self.iteration = args.iteration // args.gpu_num


        self.gan_type = args.gan_type

        self.batch_size = args.batch_size
        self.print_freq = args.print_freq // args.gpu_num
        self.save_freq = args.save_freq // args.gpu_num

        self.init_lr = args.lr
        self.ema_decay = args.ema_decay
        self.ch = args.ch

        self.dataset_path = os.path.join(self.dataset_path, 'train')
        self.label_list = [os.path.basename(x) for x in glob(self.dataset_path + '/*')]
        self.c_dim = len(self.label_list)

        self.refer_img_path = args.refer_img_path

        """ Weight """
        self.adv_weight = args.adv_weight
        self.sty_weight = args.sty_weight
        self.ds_weight = args.ds_weight
        self.cyc_weight = args.cyc_weight

        self.r1_weight = args.r1_weight
        self.gp_weight = args.gp_weight

        self.sn = args.sn

        """ Generator """
        self.style_dim = args.style_dim
        self.n_layer = args.n_layer
        self.num_style = args.num_style

        """ Discriminator """
        self.n_critic = args.n_critic

        self.img_height = args.img_height
        self.img_width = args.img_width
        self.img_ch = args.img_ch

        print()

        print("##### Information #####")
        print("# gan type : ", self.gan_type)
        print("# selected_attrs : ", self.label_list)
        print("# dataset : ", self.dataset_name)
        print("# batch_size : ", self.batch_size)
        print("# gpu num : ", self.gpu_num)
        print("# iteration : ", self.iteration)
        print("# spectral normalization : ", self.sn)

        print()

        print("##### Generator #####")
        print("# base channel : ", self.ch)
        print("# layer number : ", self.n_layer)

        print()

        print("##### Discriminator #####")
        print("# the number of critic : ", self.n_critic)

    ##################################################################################
    # Generator
    ##################################################################################

    def generator(self, x_init, style, scope="generator"):
        channel = self.ch

        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE) :
            x = conv(x_init, channels=channel, kernel=1, stride=1, use_bias=True, scope='conv_1x1')

            for i in range(self.n_layer) :
                x = pre_resblock(x, channels=channel * 2, use_bias=True, scope='down_pre_resblock_' + str(i))
                x = down_sample_avg(x)

                channel = channel * 2

            for i in range(self.n_layer // 2) :
                x = pre_resblock(x, channels=channel, use_bias=True, scope='inter_pre_resblock_' + str(i))

            for i in range(self.n_layer // 2) :
                gamma1 = fully_connected(style, channel, scope='inter_gamma1_fc_' + str(i))
                beta1 = fully_connected(style, channel, scope='inter_beta1_fc_' + str(i))

                gamma2 = fully_connected(style, channel, scope='inter_gamma2_fc_' + str(i))
                beta2 = fully_connected(style, channel, scope='inter_beta2_fc_' + str(i))

                gamma1 = tf.reshape(gamma1, shape=[gamma1.shape[0], 1, 1, -1])
                beta1 = tf.reshape(beta1, shape=[beta1.shape[0], 1, 1, -1])

                gamma2 = tf.reshape(gamma2, shape=[gamma2.shape[0], 1, 1, -1])
                beta2 = tf.reshape(beta2, shape=[beta2.shape[0], 1, 1, -1])

                x = pre_adaptive_resblock(x, channel, gamma1, beta1, gamma2, beta2, use_bias=True, scope='inter_pre_ada_resblock_' + str(i))

            for i in range(self.n_layer) :
                x = up_sample_nearest(x)

                gamma1 = fully_connected(style, channel, scope='up_gamma1_fc_' + str(i))
                beta1 = fully_connected(style, channel, scope='up_beta1_fc_' + str(i))

                gamma2 = fully_connected(style, channel // 2, scope='up_gamma2_fc_' + str(i))
                beta2 = fully_connected(style, channel // 2, scope='up_beta2_fc_' + str(i))

                gamma1 = tf.reshape(gamma1, shape=[gamma1.shape[0], 1, 1, -1])
                beta1 = tf.reshape(beta1, shape=[beta1.shape[0], 1, 1, -1])

                gamma2 = tf.reshape(gamma2, shape=[gamma2.shape[0], 1, 1, -1])
                beta2 = tf.reshape(beta2, shape=[beta2.shape[0], 1, 1, -1])

                x = pre_adaptive_resblock(x, channel // 2, gamma1, beta1, gamma2, beta2, use_bias=True, scope='up_pre_ada_resblock_' + str(i))

                channel = channel // 2

            x = conv(x, channels=self.img_ch, kernel=1, stride=1, use_bias=True, scope='return_image')

            return x

    def style_encoder(self, x_init, scope="style_encoder"):
        channel = self.ch // 2
        style_list = []
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            x = conv(x_init, channels=channel, kernel=1, stride=1, use_bias=True, scope='conv_1x1')

            for i in range(self.n_layer):
                x = pre_resblock_no_norm_relu(x, channels=channel * 2, use_bias=True, scope='down_pre_resblock_' + str(i))
                x = down_sample_avg(x)

                channel = channel * 2

            channel = channel * 2

            for i in range(self.n_layer // 2) :
                x = pre_resblock_no_norm_relu(x, channels=channel, use_bias=True, scope='down_pre_resblock_' + str(i + 4))
                x = down_sample_avg(x)

            kernel_size = int(self.img_height / np.power(2, self.n_layer + self.n_layer // 2))

            x = relu(x)
            x = conv(x, channel, kernel=kernel_size, stride=1, use_bias=True, scope='conv_g_kernel')
            x = relu(x)

            for i in range(self.c_dim) :
                style = fully_connected(x, units=64, use_bias=True, scope='style_fc_' + str(i))
                style_list.append(style)

            return style_list

    def mapping_network(self, latent_z, scope='mapping_network'):
        channel = self.ch * pow(2, self.n_layer)
        style_list = []
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            x = latent_z

            for i in range(self.n_layer + self.n_layer // 2):
                x = fully_connected(x, units=channel, use_bias=True, scope='fc_' + str(i))
                x = relu(x)

            for i in range(self.c_dim) :
                style = fully_connected(x, units=64, use_bias=True, scope='style_fc_' + str(i))
                style_list.append(style)

            return style_list # [c_dim,], style_list[i] = [bs, 64]

    ##################################################################################
    # Discriminator
    ##################################################################################

    def discriminator(self, x_init, scope="discriminator"):
        channel = self.ch
        logit_list = []
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            x = conv(x_init, channels=channel, kernel=1, stride=1, use_bias=True, scope='conv_1x1')

            for i in range(self.n_layer):
                x = pre_resblock_no_norm_lrelu(x, channels=channel * 2, use_bias=True, scope='down_pre_resblock_' + str(i))
                x = down_sample_avg(x)

                channel = channel * 2

            channel = channel * 2

            for i in range(self.n_layer // 2):
                x = pre_resblock_no_norm_lrelu(x, channels=channel, use_bias=True, scope='down_pre_resblock_' + str(i + 4))
                x = down_sample_avg(x)

            kernel_size = int(self.img_height / np.power(2, self.n_layer + self.n_layer // 2))

            x = lrelu(x, 0.2)
            x = conv(x, channel, kernel=kernel_size, stride=1, use_bias=True, scope='conv_g_kernel')
            x = lrelu(x, 0.2)

            for i in range(self.c_dim):
                logit = fully_connected(x, units=1, use_bias=True, scope='dis_logit_fc_' + str(i))
                logit_list.append(logit)

            return logit_list

    ##################################################################################
    # Model
    ##################################################################################

    def gradient_panalty(self, real, fake, real_label, scope="discriminator"):
        if self.gan_type.__contains__('dragan'):
            eps = tf.random_uniform(shape=tf.shape(real), minval=0., maxval=1.)
            _, x_var = tf.nn.moments(real, axes=[0, 1, 2, 3])
            x_std = tf.sqrt(x_var)  # magnitude of noise decides the size of local region

            fake = real + 0.5 * x_std * eps

        alpha = tf.random_uniform(shape=[self.batch_size, 1, 1, 1], minval=0., maxval=1.)
        interpolated = real + alpha * (fake - real)

        logits = tf.gather(self.discriminator(interpolated, scope=scope), real_label)


        grad = tf.gradients(logits, interpolated)[0] # gradient of D(interpolated)
        grad_norm = tf.norm(flatten(grad), axis=-1) # l2 norm

        # WGAN - LP
        GP = 0
        if self.gan_type == 'wgan-lp' :
            GP = self.gp_weight * tf.square(tf.maximum(0.0, grad_norm - 1.))

        elif self.gan_type == 'wgan-gp' or self.gan_type == 'dragan':
            GP = self.gp_weight * tf.square(grad_norm - 1.)

        return GP

    def build_model(self):

        self.ema = tf.train.ExponentialMovingAverage(decay=self.ema_decay)

        if self.phase == 'train' :
            """ Input Image"""
            img_class = Image_data(self.img_height, self.img_width, self.img_ch, self.dataset_path, self.label_list,
                                   self.augment_flag)
            img_class.preprocess()

            dataset_num = len(img_class.image)
            print("Dataset number : ", dataset_num)

            self.lr = tf.placeholder(tf.float32, name='learning_rate')
            self.ds_weight_placeholder = tf.placeholder(tf.float32, name='ds_weight')


            img_and_label = tf.data.Dataset.from_tensor_slices((img_class.image, img_class.label))

            gpu_device = '/gpu:0'
            img_and_label = img_and_label.apply(shuffle_and_repeat(dataset_num)).apply(
                map_and_batch(img_class.image_processing, self.batch_size * self.gpu_num, num_parallel_batches=16,
                              drop_remainder=True)).apply(prefetch_to_device(gpu_device, None))

            img_and_label_iterator = img_and_label.make_one_shot_iterator()

            self.x_real, label_org = img_and_label_iterator.get_next() # [bs, 256, 256, 3], [bs, 1]
            # label_trg = tf.random_shuffle(label_org)  # Target domain labels
            label_trg = tf.random_uniform(shape=tf.shape(label_org), minval=0, maxval=self.c_dim, dtype=tf.int32) # Target domain labels

            """ split """
            x_real_gpu_split = tf.split(self.x_real, num_or_size_splits=self.gpu_num, axis=0)
            label_org_gpu_split = tf.split(label_org, num_or_size_splits=self.gpu_num, axis=0)
            label_trg_gpu_split = tf.split(label_trg, num_or_size_splits=self.gpu_num, axis=0)

            g_adv_loss_per_gpu = []
            g_sty_recon_loss_per_gpu = []
            g_sty_diverse_loss_per_gpu = []
            g_cyc_loss_per_gpu = []
            g_loss_per_gpu = []

            d_adv_loss_per_gpu = []
            d_loss_per_gpu = []

            for gpu_id in range(self.gpu_num):
                with tf.device(tf.DeviceSpec(device_type="GPU", device_index=gpu_id)):
                    with tf.variable_scope(tf.get_variable_scope(), reuse=(gpu_id > 0)):

                        x_real_split = tf.split(x_real_gpu_split[gpu_id], num_or_size_splits=self.batch_size, axis=0)
                        label_org_split = tf.split(label_org_gpu_split[gpu_id], num_or_size_splits=self.batch_size, axis=0)
                        label_trg_split = tf.split(label_trg_gpu_split[gpu_id], num_or_size_splits=self.batch_size, axis=0)

                        g_adv_loss = None
                        g_sty_recon_loss = None
                        g_sty_diverse_loss = None
                        g_cyc_loss = None

                        d_adv_loss = None
                        d_simple_gp = None
                        d_gp = None

                        for each_bs in range(self.batch_size) :
                            """ Define Generator, Discriminator """
                            x_real_each = x_real_split[each_bs] # [1, 256, 256, 3]
                            label_org_each = tf.squeeze(label_org_split[each_bs], axis=[0, 1]) # [1, 1] -> []
                            label_trg_each = tf.squeeze(label_trg_split[each_bs], axis=[0, 1])

                            random_style_code = tf.random_normal(shape=[1, self.style_dim])
                            random_style_code_1 = tf.random_normal(shape=[1, self.style_dim])
                            random_style_code_2 = tf.random_normal(shape=[1, self.style_dim])

                            random_style = tf.gather(self.mapping_network(random_style_code), label_trg_each)
                            random_style_1 = tf.gather(self.mapping_network(random_style_code_1), label_trg_each)
                            random_style_2 = tf.gather(self.mapping_network(random_style_code_2), label_trg_each)

                            x_fake = self.generator(x_real_each, random_style) # for adversarial objective
                            x_fake_1 = self.generator(x_real_each, random_style_1) # for style diversification
                            x_fake_2 = self.generator(x_real_each, random_style_2) # for style diversification

                            x_real_each_style = tf.gather(self.style_encoder(x_real_each), label_org_each) # for cycle consistency
                            x_fake_style = tf.gather(self.style_encoder(x_fake), label_trg_each) # for style reconstruction

                            x_cycle = self.generator(x_fake, x_real_each_style) # for cycle consistency

                            real_logit = tf.gather(self.discriminator(x_real_each), label_org_each)
                            fake_logit = tf.gather(self.discriminator(x_fake), label_trg_each)

                            """ Define loss """
                            if self.gan_type.__contains__('wgan') or self.gan_type == 'dragan':
                                GP = self.gradient_panalty(real=x_real_each, fake=x_fake, real_label=label_org_each)
                            else:
                                GP = tf.constant([0], tf.float32)

                            if each_bs == 0 :
                                g_adv_loss = self.adv_weight * generator_loss(self.gan_type, fake_logit)
                                g_sty_recon_loss = self.sty_weight * L1_loss(random_style, x_fake_style)
                                g_sty_diverse_loss = self.ds_weight_placeholder * L1_loss(x_fake_1, x_fake_2)
                                g_cyc_loss = self.cyc_weight * L1_loss(x_real_each, x_cycle)

                                d_adv_loss = self.adv_weight * discriminator_loss(self.gan_type, real_logit, fake_logit)
                                d_simple_gp = self.adv_weight * simple_gp(real_logit, fake_logit, x_real_each, x_fake, r1_gamma=self.r1_weight, r2_gamma=0.0)
                                d_gp = self.adv_weight * GP

                            else :
                                g_adv_loss = tf.concat([g_adv_loss, self.adv_weight * generator_loss(self.gan_type, fake_logit)], axis=0)
                                g_sty_recon_loss = tf.concat([g_sty_recon_loss, self.sty_weight * L1_loss(random_style, x_fake_style)], axis=0)
                                g_sty_diverse_loss = tf.concat([g_sty_diverse_loss, self.ds_weight_placeholder * L1_loss(x_fake_1, x_fake_2)], axis=0)
                                g_cyc_loss = tf.concat([g_cyc_loss, self.cyc_weight * L1_loss(x_real_each, x_cycle)], axis=0)

                                d_adv_loss = tf.concat([d_adv_loss, self.adv_weight * discriminator_loss(self.gan_type, real_logit, fake_logit)], axis=0)
                                d_simple_gp = tf.concat([d_simple_gp, self.adv_weight * simple_gp(real_logit, fake_logit, x_real_each, x_fake, r1_gamma=self.r1_weight, r2_gamma=0.0)], axis=0)
                                d_gp = tf.concat([d_gp, self.adv_weight * GP], axis=0)


                        g_adv_loss = tf.reduce_mean(g_adv_loss)
                        g_sty_recon_loss = tf.reduce_mean(g_sty_recon_loss)
                        g_sty_diverse_loss = tf.reduce_mean(g_sty_diverse_loss)
                        g_cyc_loss = tf.reduce_mean(g_cyc_loss)

                        d_adv_loss = tf.reduce_mean(d_adv_loss)
                        d_simple_gp = tf.reduce_mean(tf.reduce_sum(d_simple_gp, axis=[1, 2, 3]))
                        d_gp = tf.reduce_mean(d_gp)

                        g_loss = g_adv_loss + g_sty_recon_loss - g_sty_diverse_loss + g_cyc_loss
                        d_loss = d_adv_loss + d_simple_gp + d_gp

                        g_adv_loss_per_gpu.append(g_adv_loss)
                        g_sty_recon_loss_per_gpu.append(g_sty_recon_loss)
                        g_sty_diverse_loss_per_gpu.append(g_sty_diverse_loss)
                        g_cyc_loss_per_gpu.append(g_cyc_loss)

                        d_adv_loss_per_gpu.append(d_adv_loss)

                        g_loss_per_gpu.append(g_loss)
                        d_loss_per_gpu.append(d_loss)

            g_adv_loss = tf.reduce_mean(g_adv_loss_per_gpu)
            g_sty_recon_loss = tf.reduce_mean(g_sty_recon_loss_per_gpu)
            g_sty_diverse_loss = tf.reduce_mean(g_sty_diverse_loss_per_gpu)
            g_cyc_loss = tf.reduce_mean(g_cyc_loss_per_gpu)
            self.g_loss = tf.reduce_mean(g_loss_per_gpu)

            d_adv_loss = tf.reduce_mean(d_adv_loss_per_gpu)
            self.d_loss = tf.reduce_mean(d_loss_per_gpu)


            """ Training """
            t_vars = tf.trainable_variables()
            G_vars = [var for var in t_vars if 'generator' in var.name]
            E_vars = [var for var in t_vars if 'encoder' in var.name]
            F_vars = [var for var in t_vars if 'mapping' in var.name]
            D_vars = [var for var in t_vars if 'discriminator' in var.name]

            if self.gpu_num == 1 :
                prev_g_optimizer = tf.train.AdamOptimizer(self.lr, beta1=0, beta2=0.99).minimize(self.g_loss, var_list=G_vars)
                prev_e_optimizer = tf.train.AdamOptimizer(self.lr, beta1=0, beta2=0.99).minimize(self.g_loss, var_list=E_vars)
                prev_f_optimizer = tf.train.AdamOptimizer(self.lr * 0.01, beta1=0, beta2=0.99).minimize(self.g_loss, var_list=F_vars)

                self.d_optimizer = tf.train.AdamOptimizer(self.lr, beta1=0, beta2=0.99).minimize(self.d_loss, var_list=D_vars)

            else :
                prev_g_optimizer = tf.train.AdamOptimizer(self.lr, beta1=0, beta2=0.99).minimize(self.g_loss, var_list=G_vars,
                                                                                                 colocate_gradients_with_ops=True)
                prev_e_optimizer = tf.train.AdamOptimizer(self.lr, beta1=0, beta2=0.99).minimize(self.g_loss,
                                                                                                 var_list=E_vars,
                                                                                                 colocate_gradients_with_ops=True)
                prev_f_optimizer = tf.train.AdamOptimizer(self.lr * 0.01, beta1=0, beta2=0.99).minimize(self.g_loss,
                                                                                                        var_list=F_vars,
                                                                                                        colocate_gradients_with_ops=True)

                self.d_optimizer = tf.train.AdamOptimizer(self.lr, beta1=0, beta2=0.99).minimize(self.d_loss,
                                                                                                 var_list=D_vars,
                                                                                                 colocate_gradients_with_ops=True)

            with tf.control_dependencies([prev_g_optimizer, prev_e_optimizer, prev_f_optimizer]):
                self.g_optimizer = self.ema.apply(G_vars)
                self.e_optimizer = self.ema.apply(E_vars)
                self.f_optimizer = self.ema.apply(F_vars)

            """" Summary """
            self.Generator_loss = tf.summary.scalar("g_loss", self.g_loss)
            self.Discriminator_loss = tf.summary.scalar("d_loss", self.d_loss)

            self.g_adv_loss = tf.summary.scalar("g_adv_loss", g_adv_loss)
            self.g_sty_recon_loss = tf.summary.scalar("g_sty_recon_loss", g_sty_recon_loss)
            self.g_sty_diverse_loss = tf.summary.scalar("g_sty_diverse_loss", g_sty_diverse_loss)
            self.g_cyc_loss = tf.summary.scalar("g_cyc_loss", g_cyc_loss)

            self.d_adv_loss = tf.summary.scalar("d_adv_loss", d_adv_loss)

            g_summary_list = [self.Generator_loss, self.g_adv_loss, self.g_sty_recon_loss, self.g_sty_diverse_loss, self.g_cyc_loss]
            d_summary_list = [self.Discriminator_loss, self.d_adv_loss]

            self.g_summary_loss = tf.summary.merge(g_summary_list)
            self.d_summary_loss = tf.summary.merge(d_summary_list)

            """ Result Image """
            def return_g_images(generator, image, code):
                x = generator(image, code)
                return x

            self.x_fake_list = []
            first_x_real = tf.expand_dims(self.x_real[0], axis=0)

            label_fix_list = tf.constant([idx for idx in range(self.c_dim)])

            for _ in range(self.num_style):
                random_style_code = tf.truncated_normal(shape=[1, self.style_dim])
                self.x_fake_list.append(tf.map_fn(
                    lambda c: return_g_images(self.generator,
                                              first_x_real,
                                              tf.gather(self.mapping_network(random_style_code), c)),
                    label_fix_list, dtype=tf.float32))

        elif self.phase == 'refer_test':
            """ Test """

            def return_g_images(generator, image, code):
                x = generator(image, code)
                return x

            self.custom_image = tf.placeholder(tf.float32, [1, self.img_height, self.img_width, self.img_ch], name='custom_image')
            self.refer_image = tf.placeholder(tf.float32, [1, self.img_height, self.img_width, self.img_ch], name='refer_image')


            label_fix_list = tf.constant([idx for idx in range(self.c_dim)])

            self.refer_fake_image = tf.map_fn(
                lambda c : return_g_images(self.generator,
                                           self.custom_image,
                                           tf.gather(self.style_encoder(self.refer_image), c)),
                label_fix_list, dtype=tf.float32)

        else :
            """ Test """

            def return_g_images(generator, image, code):
                x = generator(image, code)
                return x

            self.custom_image = tf.placeholder(tf.float32, [1, self.img_height, self.img_width, self.img_ch], name='custom_image')
            label_fix_list = tf.constant([idx for idx in range(self.c_dim)])

            random_style_code = tf.truncated_normal(shape=[1, self.style_dim])
            self.custom_fake_image = tf.map_fn(
                lambda c : return_g_images(self.generator,
                                           self.custom_image,
                                           tf.gather(self.mapping_network(random_style_code), c)),
                label_fix_list, dtype=tf.float32)



    def train(self):
        # initialize all variables
        tf.global_variables_initializer().run()

        # saver to save model
        self.saver = tf.train.Saver(max_to_keep=10)

        # summary writer
        self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_dir, self.sess.graph)

        # restore check-point if it exits
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            start_batch_id = checkpoint_counter
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            start_batch_id = 0
            counter = 1
            print(" [!] Load failed...")

        self.sample_dir = os.path.join(self.sample_dir, self.model_dir)
        check_folder(self.sample_dir)

        # loop for epoch
        start_time = time.time()
        past_g_loss = -1.
        lr = self.init_lr
        ds_w = self.ds_weight

        for idx in range(start_batch_id, self.iteration):
            if self.decay_flag :
                total_step = self.iteration
                current_step = idx
                decay_start_step = self.decay_iter

                if current_step < decay_start_step :
                    # lr = self.init_lr
                    ds_w = self.ds_weight
                else :
                    # lr = self.init_lr * (total_step - current_step) / (total_step - decay_start_step)
                    ds_w = self.ds_weight * (total_step - current_step) / (total_step - decay_start_step)

                """ half decay """
                """
                if idx > 0 and (idx % decay_start_step) == 0 :
                    lr = self.init_lr * pow(0.5, idx // decay_start_step)
                """

            train_feed_dict = {
                self.lr : lr,
                self.ds_weight_placeholder: ds_w
            }

            # Update D
            _, d_loss, summary_str = self.sess.run([self.d_optimizer, self.d_loss, self.d_summary_loss], feed_dict = train_feed_dict)
            self.writer.add_summary(summary_str, counter)

            # Update G
            g_loss = None
            if (counter - 1) % self.n_critic == 0 :
                real_images, fake_images, _, _, _, g_loss, summary_str = self.sess.run([self.x_real, self.x_fake_list,
                                                                                  self.g_optimizer, self.e_optimizer, self.f_optimizer,
                                                                                  self.g_loss, self.g_summary_loss], feed_dict = train_feed_dict)
                self.writer.add_summary(summary_str, counter)
                past_g_loss = g_loss

            # display training status
            counter += 1
            if g_loss == None :
                g_loss = past_g_loss

            print("iter: [%6d/%6d] time: %4.4f d_loss: %.8f, g_loss: %.8f" % (idx, self.iteration, time.time() - start_time, d_loss, g_loss))

            if np.mod(idx+1, self.print_freq) == 0 :
                real_image = np.expand_dims(real_images[0], axis=0)
                save_images(real_image, [1, 1],
                            './{}/real_{:07d}.jpg'.format(self.sample_dir, idx+1))

                merge_fake_x = None

                for ns in range(self.num_style) :
                    fake_img = np.transpose(fake_images[ns], axes=[1, 0, 2, 3, 4])[0]

                    if ns == 0 :
                        merge_fake_x = return_images(fake_img, [1, self.c_dim]) # [self.img_height, self.img_width * self.c_dim, self.img_ch]
                    else :
                        x = return_images(fake_img, [1, self.c_dim])
                        merge_fake_x = np.concatenate([merge_fake_x, x], axis=0)

                merge_fake_x = np.expand_dims(merge_fake_x, axis=0)
                save_images(merge_fake_x, [1, 1],
                            './{}/fake_{:07d}.jpg'.format(self.sample_dir, idx+1))

            if np.mod(counter - 1, self.save_freq) == 0:
                self.save(self.checkpoint_dir, counter)

        # save model for final step
        self.save(self.checkpoint_dir, counter)

    @property
    def model_dir(self):

        if self.sn:
            sn = '_sn'
        else:
            sn = ''

        return "{}_{}_{}_{}adv_{}sty_{}ds_{}cyc{}".format(self.model_name, self.dataset_name, self.gan_type,
                                                          self.adv_weight, self.sty_weight, self.ds_weight, self.cyc_weight,
                                                          sn)

    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name + '.model'), global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(ckpt_name.split('-')[-1])
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def test(self):
        tf.global_variables_initializer().run()
        test_files = glob('./dataset/{}/{}/*.jpg'.format(self.dataset_name, 'test')) + glob('./dataset/{}/{}/*.png'.format(self.dataset_name, 'test'))

        t_vars = tf.trainable_variables()
        G_vars = [var for var in t_vars if 'generator' in var.name or 'encoder' in var.name or 'mapping' in var.name]

        shadow_G_vars_dict = {}

        for g_var in G_vars:
            shadow_G_vars_dict[self.ema.average_name(g_var)] = g_var

        self.saver = tf.train.Saver(shadow_G_vars_dict)

        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        self.result_dir = os.path.join(self.result_dir, self.model_dir)
        check_folder(self.result_dir)

        if could_load :
            print(" [*] Load SUCCESS")
        else :
            print(" [!] Load failed...")

        # write html for visual comparison
        index_path = os.path.join(self.result_dir, 'index.html')
        index = open(index_path, 'w')
        index.write("<html><body><table><tr>")
        index.write("<th>name</th><th>input</th><th>output</th></tr>")

        for sample_file in tqdm(test_files):
            print("Processing image: " + sample_file)
            sample_image = load_test_image(sample_file, self.img_width, self.img_height, self.img_ch)
            image_path = os.path.join(self.result_dir, '{}'.format(os.path.basename(sample_file)))

            merge_x = None

            for i in range(self.num_style) :
                fake_img = self.sess.run(self.custom_fake_image, feed_dict={self.custom_image: sample_image})
                fake_img = np.transpose(fake_img, axes=[1, 0, 2, 3, 4])[0]

                if i == 0:
                    merge_x = return_images(fake_img, [1, self.c_dim]) # [self.img_height, self.img_width * self.c_dim, self.img_ch]
                else :
                    x = return_images(fake_img, [1, self.c_dim])
                    merge_x = np.concatenate([merge_x, x], axis=0)

            merge_x = np.expand_dims(merge_x, axis=0)

            save_images(merge_x, [1, 1], image_path)

            index.write("<td>%s</td>" % os.path.basename(image_path))
            index.write("<td><img src='%s' width='%d' height='%d'></td>" % (sample_file if os.path.isabs(sample_file) else (
                        '../..' + os.path.sep + sample_file), self.img_width, self.img_height))

            index.write("<td><img src='%s' width='%d' height='%d'></td>" % (image_path if os.path.isabs(image_path) else (
                        '../..' + os.path.sep + image_path), self.img_width * self.c_dim, self.img_height * self.num_style))
            index.write("</tr>")

        index.close()

    def refer_test(self):
        tf.global_variables_initializer().run()
        test_files = glob('./dataset/{}/{}/*.jpg'.format(self.dataset_name, 'test')) + glob('./dataset/{}/{}/*.png'.format(self.dataset_name, 'test'))

        refer_image = load_test_image(self.refer_img_path, self.img_width, self.img_height, self.img_ch)

        t_vars = tf.trainable_variables()
        G_vars = [var for var in t_vars if 'generator' in var.name or 'encoder' in var.name or 'mapping' in var.name]

        shadow_G_vars_dict = {}

        for g_var in G_vars:
            shadow_G_vars_dict[self.ema.average_name(g_var)] = g_var

        self.saver = tf.train.Saver(shadow_G_vars_dict)

        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        self.result_dir = os.path.join(self.result_dir, self.model_dir)
        check_folder(self.result_dir)

        self.result_dir = os.path.join(self.result_dir, 'refer_results')
        check_folder(self.result_dir)

        if could_load :
            print(" [*] Load SUCCESS")
        else :
            print(" [!] Load failed...")

        # write html for visual comparison
        index_path = os.path.join(self.result_dir, 'index.html')
        index = open(index_path, 'w')
        index.write("<html><body><table><tr>")
        index.write("<th>name</th><th>input</th><th>output</th></tr>")

        for sample_file in tqdm(test_files):
            print("Processing image: " + sample_file)
            sample_image = load_test_image(sample_file, self.img_width, self.img_height, self.img_ch)
            image_path = os.path.join(self.result_dir, '{}'.format(os.path.basename(sample_file)))

            fake_img = self.sess.run(self.refer_fake_image, feed_dict={self.custom_image: sample_image, self.refer_image: refer_image})
            fake_img = np.transpose(fake_img, axes=[1, 0, 2, 3, 4])[0]

            merge_x = return_images(fake_img, [1, self.c_dim]) # [self.img_height, self.img_width * self.c_dim, self.img_ch]
            merge_x = np.expand_dims(merge_x, axis=0)

            save_images(merge_x, [1, 1], image_path)

            index.write("<td>%s</td>" % os.path.basename(image_path))
            index.write("<td><img src='%s' width='%d' height='%d'></td>" % (sample_file if os.path.isabs(sample_file) else (
                        '../../..' + os.path.sep + sample_file), self.img_width, self.img_height))

            index.write("<td><img src='%s' width='%d' height='%d'></td>" % (image_path if os.path.isabs(image_path) else (
                        '../../..' + os.path.sep + image_path), self.img_width * self.c_dim, self.img_height))
            index.write("</tr>")

        index.close()