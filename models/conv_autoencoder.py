import tensorflow as tf
from keras.callbacks import TensorBoard

from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Input
from keras import backend as K
from keras.models import Model

from model import AbstractModel, AbstractConfig

class Config(AbstractConfig):

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
    

    def train_epoch(self, dataset, batch_size, testset=None, test_period=1):
        """Fits the model parameters to the dataset.
           Only one epoch.
        Args:
            dataset: Instance of Dataset class (see ../dataset/dataset.py)
            batch_size: Number of samples per batch.
            testset: Data on which to evaluate the model.
            test_period: Defines period after which the model is evaluated,
                         if testset is not None. 0 means never.
              
        Returns:
            Metrics like average loss, accuracy, etc..
        """
        step = 0
        for batch in dataset.batch_iter(batch_size):
            step =+ 1
            statistics_train = self.autoencoder.train_on_batch(batch, batch)
            
            # Evaluate model on testset after some steps
            if testset is not None and step%test_period==0:
                test_loss = self.validate(testset, batch_size)
                
        return "TODO"
   

    def validate(self, testset, batch_size):
        """Validates the model on the provided dataset.
        
        Args:
            testset: Instance of dataset class
            
        Returns:
            average batch-loss
        """
        avg_loss = 0.0
        n_batches = 0
        for batch in testset.batch_iter(batch_size):
            loss = self.autoencoder.test_on_batch(batch, batch)
            n_batches += 1
            avg_loss += loss
        avg_loss /= float(n_batches)
        return avg_loss
 
  
    def predict(self, datapoint):
        """Runs the model on the datapoint and produces the reconstruction.
        
        Args:
            datapoint: Datapoint that is used as input to the model.
            
        Returns:
            The recontructed datapoint.
        """
        return self.autoencoder(datapoint)


    def train_n_epochs_with_generator(self, dataset, batch_size, n_epochs=1, testset=None):
        """Fits the model parameters to the dataset by using a generator.
           Runs for multiple epochs.

        Args:
            dataset: Instance of Dataset class (see ../dataset/dataset.py) 
            batch_size: Number of samples per batch.
            n_epochs: Number of times to train on the whole dataset.
            testset: Dataset on which to evaluate the model.
              
        Returns:
            Metrics like average loss, accuracy, etc..
        """
        batches_per_epoch = dataset.get_size() / batch_size
        train_generator = dataset.keras_generator(batch_size)
        test_generator = testset.keras_generator(batch_size) if testset is not None else None
        test_n_batches = testset.get_size() / batch_size if testset is not None else None
        
        return self.autoencoder.fit_generator(
                    train_generator,
                    batches_per_epoch,
                    epochs=n_epochs,
                    validation_data=test_generator,
                    validation_steps=test_n_batches)
        
    
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
        self.autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')

                
        if self.debug:
            print("shape of input_img", K.int_shape(input_img))
            print("shape of c1", K.int_shape(c1))
            print("shape of encoded", K.int_shape(encoded))
            print("shape of d1", K.int_shape(d1))
            print("shape of u1", K.int_shape(u1))
            print("shape of decoed", K.int_shape(decoded))

model = ConvAutoenoder(Config())
        
if __name__ == '__main__':
    c = Config()
    model = ConvAutoenoder(c, debug=True)
    model.set_up_model((3,1024,1024))
    
    