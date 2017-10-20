from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Input
from keras.models import Model
from utils import Dimension

from models.abstract_model import ModelConfig
from models.abstract_keras_model import AbstractKerasModel


class ConvAutoenoder(AbstractKerasModel):

    def encode(self, datapoint):
        pass

    def decode(self, encoded_datapoint):
        pass

    def _new_model(self):
        """Creates a new convolutional autoencoder model.
        """
        input_dim = (self._config.input_dimensions.height,
                     self._config.input_dimensions.width,
                     self._config.input_dimensions.depth)

        input_layer = Input(shape=input_dim)
        conv1 = Conv2D(32, 7, strides=1, padding='same', activation='relu')(input_layer)
        conv2 = Conv2D(16, 5, strides=1, padding='same', activation='relu')(conv1)
        pool1 = MaxPooling2D(pool_size=4, padding='same')(conv2)
        conv3 = Conv2D(8, 3, strides=1, padding='same', activation='relu')(pool1)
        pool2 = MaxPooling2D(pool_size=4, padding='same')(conv3)
        conv4 = Conv2D(32, 7, strides=1, padding='same', activation='relu')(pool2)
        up1 = UpSampling2D(size=4)(conv4)
        conv5 = Conv2D(16, 7, strides=1, padding='same', activation='relu')(up1)
        up2 = UpSampling2D(size=4)(conv5)
        conv6 = Conv2D(input_dim[2], 5, strides=1, padding='same', activation='sigmoid')(up2)

        new_model = Model(input_layer, conv6)
        new_model.compile(optimizer='adadelta', loss='mean_squared_error')

        return new_model


model = ConvAutoenoder()
        
if __name__ == '__main__':
    config = ModelConfig(1, Dimension(1024, 512, 3), "./debug_logs")
    model = ConvAutoenoder()
    model.initialize(config=config)
