import keras
import os
import time


class ModelConfig(object):
    """Configuration for AbstractEncoderDecoder
    """

    batch_size = None
    input_dimensions = None
    log_dir = None

    def __init__(self, batch_size, input_dimensions, log_dir):
        self.batch_size = batch_size
        self.input_dimensions = input_dimensions
        self.log_dir = log_dir


class AbstractEncoderDecoder(object):
    """Abstract model class
  
    Each model needs to derive from this class.
    """

    _config = None
    _model = None
    _model_name = None

    def train(self, training_set, epochs, validation_set=None):
        """Fits the model parameters to the dataset.
           Only one epoch.

        Args:
            training_set: Instance of AbstractDataset
            epochs: Number of epochs to train
            validation_set: Data on which to evaluate the model.
                         
        Returns:
            Nothing
        """
        print("Starting training..")
        steps_per_epoch = training_set.get_size() / self._config.batch_size
        print("Calling fit_generator with steps_per_epoch=" + str(steps_per_epoch))
        print("log_dir: " + self._config.log_dir)
        tfb_callback = keras.callbacks.TensorBoard(log_dir=self._config.log_dir,
                                                   histogram_freq=1,
                                                   batch_size=self._config.batch_size,
                                                   write_graph=True,
                                                   write_grads=True,
                                                   write_images=True,
                                                   write_batch_performance=True)
        
        mckp_callback = keras.callbacks.ModelCheckpoint(str(os.path.join(self._config.log_dir, 'model.ckpt')),
                                                        monitor='val_loss',
                                                        verbose=0,
                                                        save_best_only=False,
                                                        save_weights_only=False,
                                                        mode='auto',
                                                        period=1)

        return self._model.fit_generator(training_set.batch_iter(),
                                         steps_per_epoch,
                                         epochs,
                                         validation_data=validation_set if validation_set is not None else None,
                                         validation_steps=len(validation_set[0]) if validation_set is not None else None,
                                         shuffle=False,
                                         callbacks=[tfb_callback, mckp_callback])

    def validate(self, validation_set):
        """Validates the model on the provided dataset.
        
        Args:
            validation_set: Instance of AbstractDataset class
            
        Returns:
            Some score to measure the performance of the model.
        """
        return self._model.evaluate_generator(validation_set.batch_iter(self._config.batch_size),
                                              validation_set.get_size())

    def predict_batch(self, batch):
        """Returns a encoded then decoded datapoint

        Args:
            batch: A numpy array of datapoints
        """
        return self._model.predict_on_batch(batch)

    def encode(self, datapoint):
        """Encode datapoint
         
        Args:
            datapoint: Datapoint.
            
        Returns:
            Encoded Datapoint.
        """
        raise NotImplementedError("Method not implemented.")

    def decode(self, encoded_datapoint):
        """Decodes a datapoint
            
        Args:
            encoded_datapoint: Is to be decoded.
            
        Returns:
            The decoded datapoint.
        """
        raise NotImplementedError("Method not implemented.")

    def _new_model(self):
        """Creates a new model.

        Returns:
            An instance of the model
        """
        raise NotImplementedError("Method not implemented.")

    def initialize(self, config=None, restore_path=None):
        """Sets up the model. This method MUST be called before anything else.
           It is like a constructor.
           TODO: This is nasty but I found no other way that works with the main.py
           
        Args:
            config: Hyperparameters of the model
            restore_path: Path to a stored model state.
                     If None, a new model will be created.
        """
        self._config = config
        self._model_name = str(type(self).__name__)
        self._config.log_dir = str(os.path.join(self._config.log_dir, self._model_name)) + "_" + str(int(time.time()))
        if restore_path is None:
            self._model = self._new_model()
            self._model.summary()
        else:
            self._model = keras.models.load_model(restore_path)
