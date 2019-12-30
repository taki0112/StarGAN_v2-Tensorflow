import tensorflow as tf
import tensorflow.contrib as tf_contrib
from utils import pytorch_xavier_weight_factor, pytorch_kaiming_weight_factor

##################################################################################
# Initialization
##################################################################################

factor, mode, uniform = pytorch_kaiming_weight_factor(uniform=False)
weight_init = tf_contrib.layers.variance_scaling_initializer(factor=factor, mode=mode, uniform=uniform)
weight_regularizer = None
weight_regularizer_fully = None


##################################################################################
# Layers
##################################################################################

# padding='SAME' ======> pad = floor[ (kernel - stride) / 2 ]
def conv(x, channels, kernel=4, stride=2, pad=0, pad_type='zero', use_bias=True, bias_init = tf.constant_initializer(0.0), sn=False, scope='conv_0'):
    with tf.variable_scope(scope):
        if pad > 0:
            h = x.get_shape().as_list()[1]
            if h % stride == 0:
                pad = pad * 2
            else:
                pad = max(kernel - (h % stride), 0)

            pad_top = pad // 2
            pad_bottom = pad - pad_top
            pad_left = pad // 2
            pad_right = pad - pad_left

            if pad_type == 'zero':
                x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])
            if pad_type == 'reflect':
                x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], mode='REFLECT')

        if sn:
            w = tf.get_variable("kernel", shape=[kernel, kernel, x.get_shape()[-1], channels], initializer=weight_init,
                                regularizer=weight_regularizer)
            x = tf.nn.conv2d(input=x, filter=spectral_norm(w),
                             strides=[1, stride, stride, 1], padding='VALID')
            if use_bias:
                bias = tf.get_variable("bias", [channels], initializer=bias_init)
                x = tf.nn.bias_add(x, bias)

        else:
            x = tf.layers.conv2d(inputs=x, filters=channels,
                                 kernel_size=kernel, kernel_initializer=weight_init,
                                 kernel_regularizer=weight_regularizer,
                                 strides=stride, use_bias=use_bias, bias_initializer=bias_init)

        return x

def fully_connected(x, units, use_bias=True, sn=False, scope='linear'):
    with tf.variable_scope(scope):
        x = flatten(x)
        shape = x.get_shape().as_list()
        channels = shape[-1]

        if sn:
            w = tf.get_variable("kernel", [channels, units], tf.float32,
                                initializer=weight_init, regularizer=weight_regularizer_fully)
            if use_bias:
                bias = tf.get_variable("bias", [units],
                                       initializer=tf.constant_initializer(0.0))

                x = tf.matmul(x, spectral_norm(w)) + bias
            else:
                x = tf.matmul(x, spectral_norm(w))

        else:
            x = tf.layers.dense(x, units=units, kernel_initializer=weight_init,
                                kernel_regularizer=weight_regularizer_fully,
                                use_bias=use_bias)

        return x

def flatten(x) :
    return tf.layers.flatten(x)

##################################################################################
# Residual-block
##################################################################################

def pre_resblock(x_init, channels, use_bias=True, sn=False, scope='resblock'):
    with tf.variable_scope(scope):
        _, _, _, init_channel = x_init.get_shape().as_list()

        with tf.variable_scope('res1'):
            x = relu(x_init)
            x = instance_norm(x)
            x = conv(x, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias, sn=sn)

        with tf.variable_scope('res2'):
            x = relu(x)
            x = instance_norm(x)
            x = conv(x, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias, sn=sn)

        if init_channel != channels :
            with tf.variable_scope('shortcut'):
                x_init = conv(x_init, channels, kernel=1, stride=1, use_bias=False, sn=sn)

        return x + x_init

def pre_resblock_no_norm_lrelu(x_init, channels, use_bias=True, sn=False, scope='resblock'):
    with tf.variable_scope(scope):
        _, _, _, init_channel = x_init.get_shape().as_list()

        with tf.variable_scope('res1'):
            x = lrelu(x_init, 0.2)
            x = conv(x, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias, sn=sn)

        with tf.variable_scope('res2'):
            x = lrelu(x, 0.2)
            x = conv(x, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias, sn=sn)

        if init_channel != channels :
            with tf.variable_scope('shortcut'):
                x_init = conv(x_init, channels, kernel=1, stride=1, use_bias=False, sn=sn)

        return x + x_init

def pre_resblock_no_norm_relu(x_init, channels, use_bias=True, sn=False, scope='resblock'):
    with tf.variable_scope(scope):
        _, _, _, init_channel = x_init.get_shape().as_list()

        with tf.variable_scope('res1'):
            x = relu(x_init)
            x = conv(x, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias, sn=sn)

        with tf.variable_scope('res2'):
            x = relu(x)
            x = conv(x, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias, sn=sn)

        if init_channel != channels :
            with tf.variable_scope('shortcut'):
                x_init = conv(x_init, channels, kernel=1, stride=1, use_bias=False, sn=sn)

        return x + x_init

def pre_adaptive_resblock(x_init, channels, gamma1, beta1, gamma2, beta2, use_bias=True, sn=False, scope='adaptive_resblock') :
    with tf.variable_scope(scope):
        _, _, _, init_channel = x_init.get_shape().as_list()
        with tf.variable_scope('res1'):
            x = relu(x_init)
            x = adaptive_instance_norm(x, gamma1, beta1)
            x = conv(x, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias, bias_init=tf.constant_initializer(1.0), sn=sn)

        with tf.variable_scope('res2'):
            x = relu(x)
            x = adaptive_instance_norm(x, gamma2, beta2)
            x = conv(x, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias, bias_init=tf.constant_initializer(1.0), sn=sn)

        if init_channel != channels :
            with tf.variable_scope('shortcut'):
                x_init = conv(x_init, channels, kernel=1, stride=1, use_bias=False, sn=sn)

        return x + x_init

##################################################################################
# Activation function
##################################################################################

def lrelu(x, alpha=0.2):
    return tf.nn.leaky_relu(x, alpha)

def relu(x):
    return tf.nn.relu(x)


##################################################################################
# Pooling & Resize
##################################################################################

def up_sample_nearest(x, scale_factor=2):
    _, h, w, _ = x.get_shape().as_list()
    new_size = [h * scale_factor, w * scale_factor]
    return tf.image.resize_nearest_neighbor(x, size=new_size)

def down_sample_avg(x, scale_factor=2):
    return tf.layers.average_pooling2d(x, pool_size=2, strides=scale_factor, padding='SAME')


##################################################################################
# Normalization function
##################################################################################

def instance_norm(x, scope='instance_norm'):
    return tf_contrib.layers.instance_norm(x,
                                           epsilon=1e-05,
                                           center=True, scale=True,
                                           scope=scope)

def adaptive_instance_norm(content, gamma, beta, epsilon=1e-5):

    c_mean, c_var = tf.nn.moments(content, axes=[1, 2], keep_dims=True)
    c_std = tf.sqrt(c_var + epsilon)

    return gamma * ((content - c_mean) / c_std) + beta

def spectral_norm(w, iteration=1):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):
        """
        power iteration
        Usually iteration = 1 will be enough
        """
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = tf.nn.l2_normalize(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = tf.nn.l2_normalize(u_)

    u_hat = tf.stop_gradient(u_hat)
    v_hat = tf.stop_gradient(v_hat)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = w / sigma
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm

##################################################################################
# Loss function
##################################################################################

def discriminator_loss(loss_func, real_logit, fake_logit):

    real_loss = 0
    fake_loss = 0

    if loss_func.__contains__('wgan') :
        real_loss = -real_logit
        fake_loss = fake_logit

    if loss_func == 'lsgan' :
        real_loss = tf.squared_difference(real_logit, 1.0)
        fake_loss = tf.square(fake_logit)

    if loss_func == 'gan' or loss_func == 'dragan' :
        real_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real_logit), logits=real_logit)
        fake_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake_logit), logits=fake_logit)

    if loss_func == 'hinge' :
        real_loss = relu(1.0 - real_logit)
        fake_loss = relu(1.0 + fake_logit)



    return real_loss + fake_loss

def generator_loss(loss_func, fake_logit):
    fake_loss = 0

    if loss_func.__contains__('wgan') :
        fake_loss = -fake_logit

    if loss_func == 'lsgan' :
        fake_loss = tf.squared_difference(fake_logit, 1.0)

    if loss_func == 'gan' or loss_func == 'dragan' :
        fake_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake_logit), logits=fake_logit)

    if loss_func == 'hinge' :
        fake_loss = -fake_logit


    return fake_loss

def simple_gp(real_logit, fake_logit, real_images, fake_images, r1_gamma=10, r2_gamma=0):
    # Used in StyleGAN

    r1_penalty = 0
    r2_penalty = 0

    if r1_gamma != 0:
        real_loss = tf.reduce_mean(real_logit)  # FUNIT = reduce_mean, StyleGAN = reduce_sum
        real_grads = tf.gradients(real_loss, real_images)[0]

        r1_penalty = r1_gamma * tf.square(real_grads)
        # FUNIT didn't use 0.5

    if r2_gamma != 0:
        fake_loss = tf.reduce_mean(fake_logit)  # FUNIT = reduce_mean, StyleGAN = reduce_sum
        fake_grads = tf.gradients(fake_loss, fake_images)[0]

        r2_penalty = r2_gamma * tf.square(fake_grads)
        # FUNIT didn't use 0.5

    return r1_penalty + r2_penalty


def L1_loss(x, y):
    loss = tf.abs(x - y)

    return loss

"""
# No use
def discriminator_loss_(loss_func, real_logit, fake_logit):

    real_loss = 0
    fake_loss = 0

    if loss_func.__contains__('wgan') :
        real_loss = -tf.reduce_mean(real_logit)
        fake_loss = tf.reduce_mean(fake_logit)

    if loss_func == 'lsgan' :
        real_loss = tf.reduce_mean(tf.squared_difference(real_logit, 1.0))
        fake_loss = tf.reduce_mean(tf.square(fake_logit))

    if loss_func == 'gan' or loss_func == 'dragan' :
        real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real_logit), logits=real_logit))
        fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake_logit), logits=fake_logit))

    if loss_func == 'hinge' :
        real_loss = tf.reduce_mean(relu(1.0 - real_logit))
        fake_loss = tf.reduce_mean(relu(1.0 + fake_logit))



    return real_loss + fake_loss

def generator_loss_(loss_func, fake_logit):
    fake_loss = 0

    if loss_func.__contains__('wgan') :
        fake_loss = -tf.reduce_mean(fake_logit)

    if loss_func == 'lsgan' :
        fake_loss = tf.reduce_mean(tf.squared_difference(fake_logit, 1.0))

    if loss_func == 'gan' or loss_func == 'dragan' :
        fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake_logit), logits=fake_logit))

    if loss_func == 'hinge' :
        fake_loss = -tf.reduce_mean(fake_logit)


    return fake_loss

def simple_gp_(real_logit, fake_logit, real_images, fake_images, r1_gamma=10, r2_gamma=0):
    # Used in StyleGAN

    r1_penalty = 0
    r2_penalty = 0

    if r1_gamma != 0:
        real_loss = tf.reduce_mean(real_logit)  # FUNIT = reduce_mean, StyleGAN = reduce_sum
        real_grads = tf.gradients(real_loss, real_images)[0]

        r1_penalty = r1_gamma * tf.reduce_mean(tf.reduce_sum(tf.square(real_grads), axis=[1, 2, 3]))
        # FUNIT didn't use 0.5

    if r2_gamma != 0:
        fake_loss = tf.reduce_mean(fake_logit)  # FUNIT = reduce_mean, StyleGAN = reduce_sum
        fake_grads = tf.gradients(fake_loss, fake_images)[0]

        r2_penalty = r2_gamma * tf.reduce_mean(tf.reduce_sum(tf.square(fake_grads), axis=[1, 2, 3]))
        # FUNIT didn't use 0.5

    return r1_penalty + r2_penalty


def L1_loss_(x, y):
    loss = tf.reduce_mean(tf.abs(x - y))

    return loss
"""