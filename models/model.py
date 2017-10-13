import keras


class ModelConfig(object):
    """Configuration for AbstractEncoderDecoder
    """

    batch_size = None
    input_dimensions = None
    checkpoints_path = None

    def __init__(self, batch_size, input_dimensions, checkpoints_path=None):
        self.batch_size = batch_size
        self.input_dimensions = input_dimensions
        self.checkpoints_path = checkpoints_path


class AbstractEncoderDecoder(object):
    """Abstract model class
  
    Each model needs to derive from this class.
    """

    _config = None
    _model = None

    def get_callbacks(self):
        callbacks = []
        callbacks.append(
            keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1, batch_size=self._config.batch_size,
                                        write_graph=True, write_grads=True, write_images=True, embeddings_freq=1,
                                        embeddings_layer_names=None, embeddings_metadata=None))

        if self._config.checkpoints_path is not None:
            callbacks.append(keras.callbacks.ModelCheckpoint(self._config.checkpoints_path, monitor='val_loss',
                                                             verbose=0, save_best_only=False, save_weights_only=False,
                                                             mode='auto', period=1))

        return callbacks

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
        steps_per_epoch = training_set.get_size() / self._config.batch_size

        return self._model.fit_generator(training_set.batch_iter(),
                                         steps_per_epoch,
                                         epochs,
                                         validation_data=validation_set.batch_iter() if validation_set is not None else None,
                                         validation_steps=validation_set.get_size() if validation_set is not None else None,
                                         shuffle=False,
                                         callbacks=self.get_callbacks())

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

    def _debug(self):
        """Debug the model
        """
        pass

    def initialize(self, config=None, restore_path=None, debug=False):
        """Sets up the model. This method MUST be called before anything else.
           It is like a constructor.
           TODO: This is nasty but I found no other way that works with the main.py
           
        Args:
            config: Hyperparameters of the model
            debug:  Debug mode.
            restore_path: Path to a stored model state.
                     If None, a new model will be created.
        """
        if debug:
            self._debug()

        self._config = config
        if restore_path is None:
            self._model = self._new_model()
        else:
            self._model = keras.models.load_model(restore_path)

