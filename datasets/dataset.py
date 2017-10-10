import random as rand


class Dataset(object):
    """Class representing a dataset
    
    Usage:
        Derive from this class and implement _load_function and 
        _preprocess_pipeline in a new file e.g. mars_rover.py.
        After implementing at the end of the file construct this object and 
        name it dataset.
        From main.py we will import it as follows: 
        from datasets.mars_rover import dataset
    """
    
    _indices = None
    _base_path = None
    _base_name = None
    _file_ending = None
    _current_batch = 0
    _current_epoch = 1
    _num_datapoints = None
    _augmentation_multiplicator = None
    
    # TODO: augmentation_multiplicator is a bit nasty, but I dont have a better solution at the moment

    def __init__(self, augmentation_multiplicator, indices=None):
        """Construct Dataset
        
        Args:
            augmentation_multiplicator: How many datapoints result from
                                        pushing one file through the 
                                        preprocess pipeline
            indices: The indices of datapoints in this dataset
        
        Returns:
            Dataset
        """
        self._num_datapoints, self._base_path, self._base_name, self._file_ending = self._download_data()
        self._augmentation_multiplicator = augmentation_multiplicator

        if indices is None:
            self._indices = \
                list(range(1, augmentation_multiplicator*self._num_datapoints+1))
            rand.shuffle(self._indices)
        else:
            self._indices = indices

    def _download_data(self):
        """Download datapoints from the source

            Args:
                None

            Returns:
                Tuple with 4 Elements: (num_datapoints, base_path, base_name, file_ending)
        """
        raise NotImplementedError()

    def _reset_batches(self):
        """Reset current_batch and re shuffle indices for a new epoch
        """
        
        self._current_batch = 0
        rand.shuffle(self._indices)
        
    def _load_function(self, file_id):
        """Load a file into a tensor
        
        Args:
            file_id: Id of a file representing a datapoint
        
        Returns:
            A tensor containing the file in the desired form
        """
        raise NotImplementedError()
        
    def _preprocess_pipeline(self):
        """List of functions that are applied to the datapoints as a 
        preprocessing step
        
        Args:
            None
            
        Returns:
            List of functions that will be applied in the order of the list.
        """
        raise NotImplementedError()
        
    def _get_data_point_id(self, batch_id):
        """Get datapoint ID from batch_id of datapoint
        
        Args:
            batch_id: The batch_id goes from 1 to 
                      num_datapoints*augmentation_multiplicator. 
                      Each augmented datapoint is represented by a batch_id.
        
        Returns:
            Datapoint ID
        """
        
        data_point_id = batch_id % self._num_datapoints
        if data_point_id == 0:
            return self._num_datapoints
        else:
            return data_point_id

    def _get_data_point_version(self, batch_id, data_point_id):
        """Get datapoint version
        
        Args:
            batch_id: Batch ID of the datapoint
            data_point_id: ID of the datapoint
            
        Returns:
            Which augmented version of the datapoint to choose.
        """
        
        return (batch_id / data_point_id) - 1
        
    def _pad_batch(self, batch, batch_size):
        """Pad a batch
        
        Args:
            batch_size: Batch size
            batch: A batch that is smaller than batch_size
        
        Returns:
            A batch that is padded to batch_size
        """
        
        diff = batch_size - len(batch)
        for i in range(diff):
            batch.append(batch[i % len(batch)])

    def batch_iter(self, batch_size):
        """Returns the next batch of the dataset
        
        Args:
            batch_size: Size of batch
            
        Returns:
            The next batch of data in a form that is ready to be consumed by 
            the model
        """
        
        while True:
            lower = min(self._current_batch*batch_size, len(self._indices))
            upper = min((self._current_batch+1)*batch_size, len(self._indices))
            
            if lower == upper:
                self._reset_batches()
                self._current_epoch = self._current_epoch + 1
                return
            
            batch_ids = self._indices[lower:upper]
            batch = []
            for batch_id in batch_ids:
                data_point_id = self._get_data_point_id(batch_id)
                
                data_point_version = self._get_data_point_version(batch_id, 
                                                                  data_point_id)
                                               
                data_point = self._load_function(data_point_id)
                
                processed_data_point = [data_point]
                
                for fun in self._preprocess_pipeline():
                    processed_data_point = map(fun, processed_data_point)
                batch.append(processed_data_point[data_point_version])
            
            if len(batch) < batch_size:
                self._pad_batch(batch, batch_size)
            
            self._current_batch = self._current_batch + 1
            yield batch
        
    def split(self, split_ratio):
        """Splits the Dataset into Test and Train 
        
        Args:
            split_ratio: Ratio of the data that should be in the train set.
            
        Returns:
            Two new Dataset objects
        """
        
        num_datapoints_train = int(split_ratio*self._num_datapoints)
        
        indices_train = self._indices[:num_datapoints_train]
        
        indices_test = self._indices[num_datapoints_train:]
        
        return (Dataset(self._augmentation_multiplicator, indices_train),
                Dataset(self._augmentation_multiplicator, indices_test))
