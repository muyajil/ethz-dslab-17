from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Input
from keras.models import Model
from utils import Dimension

from models.model import AbstractEncoderDecoder, ModelConfig


class ConvAutoenoder(AbstractEncoderDecoder):

    def encode(self, datapoint):
        pass

    def decode(self, encoded_datapoint):
        pass

    def _new_model(self):
        """Creates a new xlutional autoencoder model.
        """
        input_dim = (self._config.input_dimensions.height,
                     self._config.input_dimensions.width,
                     self._config.input_dimensions.depth)

        input_layer = Input(shape=input_dim)
        x = Conv2D(32, 32, strides=1, padding='same', activation='relu')(input_layer)
        x = Conv2D(16, 16, strides=2, padding='same', activation='relu')(x)
        x = Conv2D(16, 16, strides=4, padding='same', activation='relu')(x)
        x = Conv2D(16, 16, strides=4, padding='same', activation='relu')(x)
        x = MaxPooling2D(pool_size=4, padding='same')(x)
        x = Conv2D(8, 8, strides=1, padding='same', activation='relu')(x)
        x = MaxPooling2D(pool_size=4, padding='same')(x)
        x = Conv2D(32, 32, strides=1, padding='same', activation='relu')(x)
        x = UpSampling2D(size=4)(x)
        x = Conv2D(16, 16, strides=1, padding='same', activation='relu')(x)
        x = UpSampling2D(size=4)(x)
        x = Conv2D(16, 16, strides=1, padding='same', activation='relu')(x)
        x = Conv2D(16, 16, strides=1, padding='same', activation='relu')(x)
        x = Conv2D(input_dim[2], 8, strides=1, padding='same', activation='sigmoid')(x)

        new_model = Model(input_layer, x)
        new_model.compile(optimizer='adadelta', loss='mean_squared_error')

        return new_model


model = ConvAutoenoder()
        
if __name__ == '__main__':
    config = ModelConfig(1, Dimension(1024, 512, 3), "./debug_logs")
    model = ConvAutoenoder()
    model.initialize(config=config)
