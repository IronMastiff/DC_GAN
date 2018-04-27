import numpy as np

def scale( x, feature_range = ( -1, 1 ) ):
    # scale to ( 0, 1 )
    x = ( ( x - x.min() ) / ( 255 - x.min() ) )

    # scale to featrue_range
    min, max = feature_range
    x = x * ( max - min ) + min

    return x

class Dataset:
    def __init__( self, train, test, val_frac = 0.5, shuffle = False, scale_func = None ):
        split_idx = int( len( test['y'] ) * ( 1 - val_frac ) )
        self.test_x, self.valid_x = test['X'][:, :, :, : split_idx], test['X'][:, :, :, split_idx :]
        self.test_y, self.valid_y = test['y'][: split_idx], test['y'][split_idx :]
        self.train_x, self.train_y = train['X'], train['y']

        self.train_x = np.rollaxis( self.train_x, 3 )
        self.valid_x = np.rollaxis( self.valid_x, 3 )
        self.test_x = np.rollaxis( self.test_x, 3 )

        if scale_func is None:
            self.scaler = scale
        else:
            self.scaler = scale_func
        self.shuffle = shuffle

    def batches( self,  batch_size ):
        if self.shuffle:
            idx = np.arange( len( dataset.train_x ) )    # np.arange( number )生成0 - number的数组
            np.random.shuffle( idx )     # np.random.shuffle( arr )把arr打乱后输出
            self.train_x = self.train_x[idx]
            self.train_y = self.train_y[idx]

        n_batches = len( self.train_y ) // batch_size
        for ii in range( 0, len( self.train_y ), batch_size ):
            x = self.train_x[ii : ii + batch_size]
            y = self.train_y[ii : ii + batch_size]

            yield self.scaler( x ), y