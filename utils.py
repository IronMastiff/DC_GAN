import matplotlib.pyplot as plt
import numpy as np
import yaml

class Flag(object):
    def __init__(self, **entries):
        self.__dict__.update(entries)


def read_conf_file(conf_file):
    with open(conf_file) as f:
        FLAGS = Flag(**yaml.load(f))
    return FLAGS

def view_samples( epoch, samples, nrows, ncols, figsize = ( 5, 5 ) ):
    fig, axes = plt.subplots( figszie = figsize, nrows = nrows, ncols = ncols, sharey = True, sharex = True )
    for ax, img in zip( axes.flatten(), samples[epoch] ):
        ax.axis( 'off' )
        img = ( ( img - img.min() ) * 255 / ( img.max() - img.min() ) ).astype( np.uit8 )
        ax.set_adjustable( 'box-forced' )
        im = ax.imshow( img, aspect = 'equal' )

    plt.subplots_adjust( wspace = 0, hspace = 0 )
    plt.show()
    return fig, axes