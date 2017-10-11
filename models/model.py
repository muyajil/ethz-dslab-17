class AbstractModel(object):
    """Abstract model class
  
    Each model needs to derive from this class.
    """
  
    def train_epoch(self, dataset, batch_size, testset=None, test_period=None):
        """Fits the model parameters to the dataset.
           Only one epoch.

        Args:
            dataset: Instance of dataset class (see ../dataset/dataset.py)
            batch_size: Number of samples per batch.
            testset: Data on which to evaluate the model.
            test_period: Defines period after which the model is evaluated,
                         if testset is not None.
                         
        Returns:
            Metrics like average loss, accuracy, etc..
        """
        raise NotImplementedError("Method not implemented.")
        
      
    def validate(self, dataset):
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

  
    def __init__(self, config, debug=False, restore=None):
        """Initialization of the model.
      
        Args:
            config: Parameters that are model-specific.
            debug: If True, only two batches will be used.
            restore: Path to a stored model state.
                     If None, a new model will be created.
        """
        self.debug = debug
        self.config = config
      
        if restore is None:
            self._new_model(config)
        else:
            self._restore_model(restore)
            