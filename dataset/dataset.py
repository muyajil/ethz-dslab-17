class Dataset(object):
    """ Class representing a dataset
    """
    
    base_path = None
    base_name = None
    preprocess_pipeline = None
    
    def __init__(self, base_path, base_name, preprocess_pipeline, num_datapoints):
        """ Construct Dataset
        
        Args:
            base_path: Path where the data files are
            base_name: Base name of the files. The files should be numbered 1 ... after the base_name
            preprocess_pipeline: List of functions that preprocess (augment, transform) datapoints
            num_datapoints: Number of datapoints
        
        Returns:
            Dataset
        """
        self.base_name = base_name
        self.base_path = base_path
        self.preprocess_pipeline = preprocess_pipeline
        
    def get_next_batch(self):
        """ Returns the next batch of the dataset
        
        Args:
            None
            
        Returns:
            Iterator to the next batch of the data
        """
        raise NotImplementedError("Method not implemented.")
        
    def split(self, split_ratio):
        """ Splits the Dataset into Test and Train 
        
        Args:
            split_ratio: Ratio of the data that should be in the test set.
            
        Returns:
            Two new Dataset objects
        """
        
    def 