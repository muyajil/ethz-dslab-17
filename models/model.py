class AbstractConfig(object):

    # Image dimensions
    img_channels = None
    img_width = None
    img_height = None

class AbstractModel(object):
    """Abstract model class
  
    Each model needs to derive from this class.
    """
  
    def train_epoch(self, dataset, batch_size, image_dim, testset=None, test_period=None):
        """Fits the model parameters to the dataset.
           Only one epoch.

        Args:
            dataset: Instance of dataset class (see ../dataset/dataset.py)
            batch_size: Number of samples per batch.
            testset: Data on which to evaluate the model.
            test_period: Defines period after which the model is evaluated,
                         if testset is not None.
                         
        Returns:
            Nothing
        """
        raise NotImplementedError("Method not implemented.")

    def validate(self, dataset, batch_size, image_dim):
        """Validates the model on the provided dataset.
        
        Args:
            dataset: Instance of dataset class
            
        Returns:
            Some score to measure the performance of the model.
        """
        raise NotImplementedError("Method not implemented.")

    def predict(self, datapoint):
        """Runs the model on the datapoint and produces the reconstruction.
        
        Args:
            datapoint: Datapoint that is used as input to the model.
            
        Returns:
            The recontructed datapoint.
        """
        raise NotImplementedError("Method not implemented.")

    def predict_batch(self, batch):
        raise NotImplementedError()

    def compress(self, datapoint):
        """First converts the given datapoint to it's latent representation,
           and then to a transmittable form (e.g., to binary).
         
        Args:
            datapoint: Datapoint that is compressed.
            
        Returns:
            The encoding of the datapoint in a transmittable form.
        """
        raise NotImplementedError("Method not implemented.")

   
    def decompress(self, compressed_datapoint):
        """First decodes compressed_datapoint to the latent space
        representation, and then to the reconstructed datapoint.
            
        Args:
            compressed_datapoint: Is to be decoded and decompressed.
            
        Returns:
            The reconstructed datapoint.
        """
        raise NotImplementedError("Method not implemented.")


    def _new_model(self, config):
        """Creates a new model.
      
        Args:
            config: Parameters that are model-specific.
        """
        raise NotImplementedError("Method not implemented.")

    
    def _restore_model(self, path):
        """Restore a stored model.
      
        Args:
            path: Path to the stored model state.
        """
        raise NotImplementedError("Method not implemented.")


    def set_up_model(self, input_dimensions, restore=None):
        """Sets up the model. This method MUST be called before anything else.
           It is like a constructor.
           TODO: This is nasty but I found no other way that works with the main.py
           
        Args:
            input_dimensions: Dimensions of the data points.
            restore: Path to a stored model state.
                     If None, a new model will be created.
        """
        self.config.img_height, self.config.img_width, self.config.img_channels = input_dimensions
        if restore is None:
            self._new_model(self.config)
        else:
            self._restore_model(restore)

    def save_model(self):
        raise NotImplementedError()
  
    def __init__(self, config, debug=False):
        """Initialization of the model.
      
        Args:
            config: Parameters that are model-specific.
            debug:  Debug mode.
        """
        self.debug = debug
        self.config = config
