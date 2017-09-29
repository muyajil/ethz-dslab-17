from random import shuffle

class Dataset(object):
    """Class representing a dataset
    """
    
    base_path = None
    base_name = None
    preprocess_pipeline = None
    indices = None
    batch_size = None
    current_batch = 0
    load_function = None
    
    def __init__(self, base_path, 
                base_name, preprocess_pipeline, 
                num_datapoints, batch_size,
                load_function):
        """Construct Dataset
        
        Args:
            base_path: Path where the data files are
            base_name: Base name of the files. The files should be numbered 1 ... after the base_name
            preprocess_pipeline: List of functions that preprocess (augment, transform) datapoints
            num_datapoints: Number of datapoints
            batch_size: Number of datapoints per batch
            load_function: Function that loads the data into the desired form from a file
        
        Returns:
            Dataset
        """
        self.base_name = base_name
        self.base_path = base_path
        self.preprocess_pipeline = preprocess_pipeline
        self.batch_size = batch_size
        self.indices = shuffle(list(range(1, num_datapoints+1)))
        self.load_function = load_function
        
    def get_next_batch(self):
        """Returns the next batch of the dataset
        
        Args:
            None
            
        Returns:
            The next batch of data
            The batch_size if forced even if augmentation is used, if so we use fewer datapoints per batch
        """
        self.current_batch = self.current_batch + 1
        
        raise NotImplementedError("Method not implemented.")
        
    def split(self, split_ratio):
        """Splits the Dataset into Test and Train 
        
        Args:
            split_ratio: Ratio of the data that should be in the test set.
            
        Returns:
            Two new Dataset objects
        """
        
        raise NotImplementedError("Method not implemented.")