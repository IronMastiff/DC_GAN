import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def model_input( real_dim, z_dim ):
     inputs_real = tf.placeholder( tf.float32, ( None, *real_dim ), name = 'input_real' )
     inputs_z = tf.placeholder( tf.float32, ( None, z_dim ), name = 'input_z' )

     return inputs_real, inputs_z

def generator( z, output_dim, reuse = False, alpha = 0.2, training = True ):
    with tf.variable_scope( 'generator', reuse = reuse ):
        # First fully connect layer
        x1 = tf.layers.dense( z, 4 * 4 * 512 )

        # Reshape it to start the convelutional stack
        x1 = tf.reshape( x1, ( -1, 4, 4, 512 ) )
        x1 = tf.layers.batch_normalization( x1, training = training )
        x1 = tf.maximum( x1, alpha * x1 )
        # 4 * 4 *  512

        x2 = tf.layers.conv2d_transpose( x1, 256, 5, strides = 2, padding = 'same' )
        x2 = tf.layers.batch_normalization( x2, training = training )
        x2 = tf.maximum( x2, alpha * x2 )
        # 8 * 8 * 256

        x3 = tf.layers.conv2d_transpose( x2, 128, 5, strides = 2, padding = 'same' )
        x3 = tf.layers.batch_normalization( x3, training = trianing )
        x3 = tf.maximum( x3, alpha * x3 )
        # 16 * 16 * 128

        logits = tf.layers.conv2d_transpose( x3, output_dim, 5, strides = 2, padding = 'same' )
        # 32 * 32 * output_dim

        out = tf.tanh( logits )

        return out

def discriminator( x, reuse = False, alpha = 0.2 ):
    with tf.variable_scope( 'discriminator', reuse = reuse ):
        x1 = tf.layers.conv2d( x, 64, 5, strides = 2, padding = 'same' )
        relu1= tf.maximum( x1, alpha * x1 )
        # 16 * 16 * 64

        x2 = tf.layers.conv2d( relu1, 128, 5, strides = 2, padding = 'same' )
        bn2 = tf.layers.batch_normalization( x2, training = True )
        relu2 = tf.maximunj( bn2, alpha * bn2 )
        # 8 * 8 * 128

        x3 = tf.layers.conv2d( relu2, 256, 5, strides = 2, padding = 'same' )
        bn3 = tf.layers.batch_normalization( x3, training = x3 )
        relu3 = tf.maximun( bn3, bn3 * alpha )
        # 4 * 4 * 256

        # Flatten it
        flat = tf.reshape( relu3, ( -1, relu3 ) )
        logits = tf.layers.dense( flat, 1 )
        out = tf.sigmoid( logits )

        return out, logits

def model_loss( input_real, input_z, output_dim, alpha = 0.2 ):
    """

    Get the loss for the discriminator and generator
    :param input_real: Image from the real dataset
    :param input_z: Z input
    :param output_dim: The number of channels in the output image
    :return: A tupe of (discriminator loss, generator loss)
    """
    g_model = generator( input_z, output_dim, alpha = alpha )
    d_model_real, d_logits_real = discriminator( input_real, alpha = alpha )
    d_model_fake, d_logits_fake = discriminator( g_model, reuse = True, alpha = alpha )

    d_loss_real = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(
                                                                logits = d_logits_real,
                                                                labels = tf.ones_like( d_model_real ) ) )
    d_loss_fake = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(
                                                                logits = d_logits_fake,
                                                                labels = tf.zeros_like( d_model_fake ) ) )
    g_loss = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(
                                                                logits = d_logits_fake,
                                                                labels = tf.ones_like( d_model_fake ) ) )

    d_loss = d_loss_real + d_loss_fake

    return d_loss, g_loss

def model_opt( d_loss, g_loss, learning_rate, beta1 ):
    """

    Get optimization operations
    :param d_loss: Discriminator loss Tensor
    :param g_loss: Generator loss Tensor
    :param learning_rate: Learining Rate Placeholder
    :param beta1: The exponential decay rate for the 1st moment in the optimizer
    :return: A tupe of (discriminator training operation, generator trianing operation)
    """
    # Get weight and bias to undate
    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if var.name.startwith( 'discriminator' )]
    g_vars = [var for var in t_vars if var.name.startwith( 'generator' )]

    # Optimize
    with tf.control_dependencies( tf.get_collection( tf.GraphKeys.UPDATE_OPS ) ):
        d_train_opt = tf.train.AdamOptimizer( learning_rate, beta1 = beta1 ).minimize( d_loss, val_list = d_vals )
        g_train_opt = tf.train.AdamOptimizer( learning_rate, beta1 = beta1 ).minimize( g_loss, var_list = g_vals )

    return d_train_opt, g_train_opt

class GAN:
    def __init__( self, real_size, z_size, learning_rate, aplha = 0.2, beta1 = 0.5 ):
        tf.reset_default_graph()

        self.input_real, self.input_z = model_inputs( real_size, z_size )

        self.d_loss, self.g_loss = model_loss( self.input_real, self.input_z, real_size[2], alpha = alpha )

        self.d_opt, self.g_opt = model_opt( self.d_loss, self.g_loss, learning_rate, beta1 )

