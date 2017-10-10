import tensorflow as tf
from keras.callbacks import TensorBoard

from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Input
from keras import backend as K
from keras.models import Model

from model import AbstractModel

class Config(object):
    batch_size = 200
    epochs = 2
    
    # Image dimensions
    img_channels = 3
    img_widht = 1024
    img_height = 1024
    
    # Conv layer
    c1_channels = 16
    c1_size = 7, 7
    c1_strides = 1, 1
    
    # Pooling
    p1_size = 4, 4
    
    # Conv layer
    d1_channels = 16
    d1_size = 7, 7
    d1_strides = 1, 1
    
    # Upsampling
    u1_size = p1_size
    
    # Conv layer
    d2_channels = 3
    d2_size = 7, 7
    d2_strides = 1, 1
    
class ConvAutoenoder(AbstractModel):
    
    def train(self, dataset, testset=None):
        """Fits the model parameters to the dataset.

        Args:
            dataset: Instance of dataset class (see ../dataset/dataset.py)
            testset: Data on which to evaluate the loss at the end of each epoch.
              
        Returns:
            Metrics like average loss, accuracy, etc..
        """
        self.autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')
        self.autoencoder.fit(dataset, dataset, epochs=self.config.epochs,
            batch_size=self.config.batch_size, shuffle=True,
            validation_data=None, verbose=1)
        # TODO    callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])

    
    def _new_model(self, config):
        """Creates a new convolutional autoencoder model.
      
        Args:
            config: Model parameters.
        """
        input_img = Input(shape=(config.img_channels,
                                config.img_widht,
                                config.img_height))

        c1 = Conv2D(
                config.c1_channels,
                config.c1_size,
                strides=config.c1_strides,
                padding='same',
                data_format='channels_first',
                activation='relu')(input_img)

        encoded = MaxPooling2D(
                pool_size=config.p1_size,
                padding='same',
                data_format='channels_first')(c1)
                
        d1 = Conv2D(
                config.d1_channels,
                config.d1_size,
                strides=config.d1_strides,
                padding='same',
                data_format='channels_first',
                activation='relu')(encoded)
                
        u1 = UpSampling2D(size=config.u1_size, data_format='channels_first')(d1)
        
        decoded = Conv2D(
                config.d2_channels,
                config.d2_size,
                strides=config.d2_strides,
                padding='same',
                data_format='channels_first',
                activation='sigmoid')(u1)
                
        self.autoencoder = Model(input_img, decoded)
                
        if self.debug:
            print("shape of input_img", K.int_shape(input_img))
            print("shape of c1", K.int_shape(c1))
            print("shape of encoded", K.int_shape(encoded))
            print("shape of d1", K.int_shape(d1))
            print("shape of u1", K.int_shape(u1))
            print("shape of decoed", K.int_shape(decoded))


        
if __name__ == '__main__':
    c = Config()
    model = ConvAutoenoder(c, debug=True)
    
    