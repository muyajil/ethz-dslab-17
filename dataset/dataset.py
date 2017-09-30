import random as rand
import os

class Dataset(object):
    """Class representing a dataset
    """
    
    config = None
    indices = None
    current_batch = 0
    
    def __init__(self, config, indices=None):
        """Construct Dataset
        
        Args:
            config: A config of a datasource. The following fields must be provided:
                config.base_path
                config.base_name
                config.preprocess_pipeline
                config.num_datapoints
                config.load_function
        
        Returns:
            Dataset
        """
        self.config = config
        
        if indices is None:
            self.indices = rand.shuffle(range(1, config.num_datapoints+1))
        else:
            self.indices = indices
    
    def reset_batches(self):
        self.current_batch = 0
        self.indices = rand.shuffle(range(1, self.config.num_datapoints+1))

    def get_next_batch(self, batch_size):
        """Returns the next batch of the dataset
        
        Args:
            None
            
        Returns:
            The next batch of data
            The batch_size if forced even if augmentation is used, if so we use fewer datapoints per batch
        """
        lower = self.current_batch*batch_size
        upper = (self.current_batch+1)*batch_size
        batch_ids = self.indices[lower:upper]
        batch = []
        for data_point_id in batch_ids:
            data_point_file = os.path.join(self.config.base_name + data_point_id, self.config.base_path)
            data_point = self.config.load_function(data_point_file)
            processed_data_point = [data_point]
            for fun in self.config.preprocess_pipeline:
                processed_data_point = map(fun, processed_data_point)
            batch.append(rand.choice(processed_data_point))
        
        return batch
        
    def split(self, split_ratio):
        """Splits the Dataset into Test and Train 
        
        Args:
            split_ratio: Ratio of the data that should be in the test set.
            
        Returns:
            Two new Dataset objects
        """
        
        num_datapoints_split = int(split_ratio*self.config.num_datapoints)
        
        split_1 = self.indices[:num_datapoints_split]
        config_1 = self.config
        config_1.num_datapoints = num_datapoints_split
        
        split_2 = self.indices[num_datapoints_split:]
        config_2 = self.config
        config_2.num_datapoints = self.config.num_datapoints - num_datapoints_split
        
        return Dataset(config_1, split_1), Dataset(config_2, split_2)
        