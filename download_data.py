from urllib.request import urlretrieve
from tqdm import tqdm
from os.path import join
from os.path import isfile

def download_data( data_dir, train_data, test_data ):
    class DLProgress(tqdm):
        last_block = 0

        def hook( self, block_num=1, block_size=1, total_size=None ):
            self.total = total_size
            self.update( ( block_num - self.last_block ) * block_size )
            self.last_block = block_num

    if not isfile( data_dir + '/' + train_data ):
        with DLProgress( unit = 'B', unit_scale = True, miniters = 1, desc = 'SVHN Training Set' ) as pbar:
            urlretrieve(
                'http://ufldl.stanford.edu/housenumbers/train_32x32.mat',
                data_dir + '/' + 'train_32x32.mat',
                pbar.hook
            )

    if not isfile( data_dir + '/' + test_data ):
        with DLProgress( unit = 'B', unit_scale = True, miniters = 1, desc = 'SVHN Testing Set' ) as pbar:
            urlretrieve(
                'http://ufldl.stanford.edu/housenumbers/test_32x32.mat',
                data_dir + '/' + 'test_32x32.mat',
                pbar.hook
            )