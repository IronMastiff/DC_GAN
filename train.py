import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

from download_data import download_data as download_data

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument( '-c', '--conf', default='conf/style.yml', help='the path to the conf file' )
    return parser.parse_args()

def main( FLAGS ):
    if not os.path.isdir( FLAGS.data_dir ):
        os.makedirs( FLAGS.data_dir )
    download_data( FLAGS.data_dir, FLAGS.train_data, FLAGS.test_data )

    train_data = loadmat( FLAGS.data_dir + FLAGS.train_data )
    test_data = loadmat( FLAGS.data_dir + FLAGS.test_data )





if __name__ == '__main__':
    args = parse_args()
    FLAGS = utils.read_conf_file( args.conf )
    main( FLAGS )