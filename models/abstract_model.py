import os
import time

class ModelConfig(object):
    """Configuration for AbstractModel
    """

    batch_size = None
    input_dimensions = None
    log_dir = None

    def __init__(self, batch_size, input_dimensions, log_dir):
        self.batch_size = batch_size
        self.input_dimensions = input_dimensions
        self.log_dir = log_dir
        
class AbstractModel(object):
    """Abstract model class
  
    Each model needs to derive from this class.
    """
    
    _config = None
    _model = None
    _model_name = None
    
    def train(self, training_set, epochs, validation_set=None):
        """Fits the model parameters to the dataset.

        Args:
            training_set: Instance of AbstractDataset
            epochs: Number of epochs to train
            validation_set: Data on which to evaluate the model.

        """
        raise NotImplementedError("Method not implemented.")

   
    def validate(self, validation_set):
        """Validates the model on the provided dataset.
        
        Args:
            validation_set: Instance of AbstractDataset class
            
        Returns:
            Some score to measure the performance of the model.
        """
        raise NotImplementedError("Method not implemented.")


    def predict_batch(self, batch):
        """Returns a encoded then decoded datapoint

        Args:
            batch: A numpy array of datapoints
        """
        raise NotImplementedError("Method not implemented.")


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
        
    
    def _restore_model(self, restore_path):
        """Restores a model stored at restore_path
        
        Args:
            restore_path: Path where the model is stored.
            
        Returns:
            An instance of the model.
        """

    def _print_model_summary(self):
        """ Prints the model structure and other information.
        """
        pass


    def initialize(self, config=None, restore_path=None):
        """Sets up the model. This method MUST be called before anything else.
           It is like a constructor.

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
            self._print_model_summary()
        else:
            self._model = self._restore_model(restore_path)
