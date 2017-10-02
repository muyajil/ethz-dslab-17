import random as rand
import os

class Dataset(object):
    """Class representing a dataset
    """
    
    config = None
    indices = None
    current_batch = 0
    current_epoch = 1
    
    def __init__(self, config, indices=None):
        """Construct Dataset
        
        Args:
            config: A DatasetConfig object
            indices: The indices of datapoints in this dataset
        
        Returns:
            Dataset
        """
        self.config = config
        
        if indices is None:
            self.indices = list(range(1, config.augmentation_multiplicator*config.num_datapoints+1))
            rand.shuffle(self.indices)
        else:
            self.indices = indices
    
    def reset_batches(self):
        self.current_batch = 0
        rand.shuffle(self.indices)
        
    def get_data_point_id(self, batch_id):
        data_point_id = batch_id % self.config.num_datapoints
        if data_point_id == 0:
            return self.config.num_datapoints
        else:
            return data_point_id

    def get_data_point_version(self, batch_id, data_point_id):
        return (batch_id / data_point_id) - 1
        
    def pad_batch(self, batch, batch_size):
        diff = batch_size - len(batch)
        for i in range(diff):
            batch.append(batch[i % len(batch)])

    def batch(self, batch_size):
        """Returns the next batch of the dataset
        
        Args:
            batch_size: Size of batch
            
        Returns:
            The next batch of data in a form that is ready to be consumed by the model
        """
        while True:
            lower = min(self.current_batch*batch_size, self.config.num_datapoints)
            upper = min((self.current_batch+1)*batch_size, self.config.num_datapoints)
            
            if lower == upper: # Epoch is done, start the next one
                self.reset_batches()
                self.current_epoch = self.current_epoch + 1
                return
            
            batch_ids = self.indices[lower:upper]
            batch = []
            for batch_id in batch_ids:
                data_point_id = self.get_data_point_id(batch_id)
                data_point_version = self.get_data_point_version(batch_id, data_point_id)
                data_point_file = os.path.join(self.config.base_name + data_point_id, self.config.base_path)
                data_point = self.config.load_function(data_point_file)
                processed_data_point = [data_point]
                for fun in self.config.preprocess_pipeline:
                    processed_data_point = map(fun, processed_data_point)
                batch.append(processed_data_point[data_point_version])
            
            if len(batch) < batch_size:
                self.pad_batch(batch, batch_size)
            
            self.current_batch = self.current_batch + 1
            yield batch
        
    def split(self, split_ratio):
        """Splits the Dataset into Test and Train 
        
        Args:
            split_ratio: Ratio of the data that should be in the train set.
            
        Returns:
            Two new Dataset objects
        """
        
        num_datapoints_split = int(split_ratio*self.config.num_datapoints)
        
        indices_train = self.indices[:num_datapoints_split]
        config_train = self.config
        config_train.num_datapoints = num_datapoints_split
        
        indices_test = self.indices[num_datapoints_split:]
        config_test = self.config
        config_test.num_datapoints = self.config.num_datapoints - num_datapoints_split
        
        return Dataset(config_train, indices_train), Dataset(config_test, indices_test)
        