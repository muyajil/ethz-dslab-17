from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Input
from keras.models import Model

from models.model import AbstractEncoderDecoder, ModelConfig


class ConvAutoenoder(AbstractEncoderDecoder):

    def encode(self, datapoint):
        pass

    def decode(self, encoded_datapoint):
        pass

    def _new_model(self):
        """Creates a new convolutional autoencoder model.
        """
        input_dim = (self._config.input_dimensions.height, self._config.input_dimensions.width)
        input_layer = Input(shape=input_dim)
        # TODO: Now the problem is that we have 2D images, i.e. we have no channels, the model should be adapted. Everything up to here is done, only the layers must be adapted.
        conv1 = Conv2D(4, 16, strides=1, padding='same', activation='relu')(input_layer)

        encoding = MaxPooling2D(pool_size=2, padding='same')(conv1)

        conv2 = Conv2D(4, 16, strides=1, padding='same', activation='relu')(encoding)

        u1 = UpSampling2D(size=2)(conv2)
        
        decoded = Conv2D(1, 16, strides=1, padding='same', activation='sigmoid')(u1)

        new_model = Model(input_layer, decoded)
        new_model.compile(optimizer='adadelta', loss='mean_squared_error')

        return new_model


model = ConvAutoenoder()
        
if __name__ == '__main__':
    config = ModelConfig(64, (1024, 1024, 3))
    model = ConvAutoenoder()
    model.initialize(config=config, debug=True)
