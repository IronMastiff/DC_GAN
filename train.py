import tensorflow as tf
import os
import matplotlib.pyplot as plt
import argparse
import numpy as np
from scipy.io import loadmat

from download_data import download_data as download_data
from net import GAN, generator
import utils


'''--------Hyperparameters--------'''
real_size = ( 32, 32, 3 )
z_size = 100
learning_rate = 0.001

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument( '-c', '--conf', default='conf/style.yml', help='the path to the conf file' )
    return parser.parse_args()

def main( FLAGS ):
    if not os.path.isdir( FLAGS.data_dir ):
        os.makedirs( FLAGS.data_dir )
    download_data( FLAGS.data_dir, FLAGS.train_data, FLAGS.test_data )

    net = GAN( real_size, z_size, learning_rate, alpha = FLAGS.alpha, beta1 = FLAGS.beta1)

    '''--------Load data--------'''
    train_data = loadmat( FLAGS.data_dir + FLAGS.train_data )
    test_data = loadmat( FLAGS.data_dir + FLAGS.test_data )

    '''--------Build net--------'''
    saver = tf.train.Saver()
    sample_z = np.random.uniform( -1, 1, size = ( 72, z_size ) )

    samples, losses = [], []
    steps = 0

    with tf.Session() as sess:
        sess.run( tf.global_variables_initializer() )
        for e in range( FLAGS.epochs ):
            for x, y in dataset.batches( batch_size ):
                steps += 1

                # Sample random noise for G
                batch_z = np.random.uniform( -1, 1, size = ( batch_size, z_size ) )

                # Run optimizers
                _ = sess.run( net.d_opt, feed_dict = { net.input_real: x, net.input_z: batch_z} )
                _ = sess.run( net.g_opt, feed_dict = { net.input_z: batch_z, net.input_real: x} )

                if steps % print_every == 0:
                    # At the end of each epoch, get the losses and print them out
                    train_loss_d = net.d_loss.eval( {net.input_z: batch_z, net.input_real: x} )
                    train_loss_g = net.g_loss.eval( {net.input_z: batch_z} )

                    print( 'Epoch {}/{}...'.format( e + 1, FLAGS.epochs ),
                           'Discriminator Loss: {:.4f}'.format( train_loss_g) )
                    # Save losses to view after traning
                    losses.append( ( train_loss_d, train_loss_g ) )

                if steps % show_every == 0:
                    gen_samples = sess.run( generator( net.input_z, 3, reuse = True, training = False ),
                                            feed_dict = {net.input_z: sample_z} )
                    samples.append( gen_samples )
                    _ = utils.view_samples( -1, samples, 6, 12, figsize = figsize )
                    plt.show()

            saver.save( sess, './checkpoints/generator.ckpt' )

        with open( 'samples.pkl', 'wb' ) as f:
            pkl.dump( samples, f )


        fig, ax = plt.subplot()
        losses = np.array( losses )
        plt.plot( losses.T[0], label = 'Discriminator', alpha = 0.5 )
        plt.plot( losses.T[1], label = 'Generator', alpha = 0.5 )
        plt.title( 'Training Losses' )
        plt.legend()
        plt.show()



if __name__ == '__main__':
    args = parse_args()
    FLAGS = utils.read_conf_file( args.conf )
    main( FLAGS )