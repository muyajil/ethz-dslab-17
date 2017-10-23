import random as rand
import numpy as np
import os
import copy


class DatasetConfig(object):
    base_path = None
    augmentation_multiplicator = None
    num_datapoints = None
    input_dimensions = None
    batch_size = None

    def __init__(self, augmentation_multiplicator):
        self.augmentation_multiplicator = augmentation_multiplicator

    def initialize(self, base_path, input_dimensions, batch_size):
        self.base_path = base_path
        self.batch_size = batch_size
        self.input_dimensions = input_dimensions
        self.num_datapoints = len(os.listdir(self.base_path))


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
    _current_batch = 0
    _current_epoch = 1
    _config = None

    def _reset_batches(self):
        """Reset current_batch and re shuffle indices for a new epoch
        """
        
        self._current_batch = 0
        rand.shuffle(self._indices)
        
    def _load_function(self, file_name):
        """Load a file into a tensor
        
        Args:
            file_name: File name of file representing a datapoint
        
        Returns:
            A tensor containing the file in the desired form
        """
        raise NotImplementedError()
        
    def _preprocess_pipeline(self):
        """List of functions that are applied to the datapoints as a 
        preprocessing step
            
        Returns:
            List of functions that will be applied in the order of the list.
        """
        raise NotImplementedError()

    def _crop_input(self, datapoint):
        """This function crops the possibly variable data dimensions to the one specified in the config
        This function is always called after preprocessing a datapoint.

        Args:
            datapoint: A datapoint from the dataset

        Returns:
            A cropped version of datapoint that fits into the input of the model that initialized this class.
        """
        raise NotImplementedError()

    def get_debug_dataset(self):
        """Generates a dataset for debugging

            Returns:
                A Dataset for Debugging
        """

        debug_datapoints = self._config.batch_size*3 + 1
        debug_config = copy.deepcopy(self._config)
        debug_config.num_datapoints = debug_datapoints
        debug_indices = self._indices[:debug_datapoints]
        debug_dataset = type(self)()
        debug_dataset.initialize(debug_config, indices=debug_indices)
        return debug_dataset
        
    def _get_data_point_id(self, batch_id):
        """Get datapoint ID from batch_id of datapoint
        
        Args:
            batch_id: The batch_id goes from 1 to 
                      num_datapoints*augmentation_multiplicator. 
                      Each augmented datapoint is represented by a batch_id.
        
        Returns:
            Datapoint ID
        """
        
        data_point_id = batch_id % self._config.num_datapoints
        if data_point_id == 0:
            return self._config.num_datapoints
        else:
            return data_point_id

    def _get_data_point_version(self, batch_id):
        """Get datapoint version
        
        Args:
            batch_id: Batch ID of the datapoint

        Returns:
            Which augmented version of the datapoint to choose.
        """
        num_files = len(os.listdir(self._config.base_path))
        return int(batch_id / num_files)
        
    def _pad_batch(self, batch):
        """Pad a batch
        
        Args:
            batch: A batch that is smaller than batch_size
        
        Returns:
            A batch that is padded to batch_size
        """
        
        diff = self._config.batch_size - len(batch)
        for i in range(diff):
            batch.append(batch[i % len(batch)])

    def batch_iter(self, stop_after_epoch=False):
        """Returns the next batch of the dataset

        Returns:
            The next batch of data in a form that is ready to be consumed by 
            the model
        """
        
        while True:
            lower = min(self._current_batch*self._config.batch_size, len(self._indices)-1)
            upper = min((self._current_batch+1)*self._config.batch_size, len(self._indices)-1)
            
            if lower == upper:
                self._reset_batches()
                self._current_epoch = self._current_epoch + 1
                if stop_after_epoch:
                    return
                continue
            
            batch_ids = self._indices[lower:upper]
            batch = []
            for batch_id in batch_ids:
                data_point_id = self._get_data_point_id(batch_id)
                data_point_version = self._get_data_point_version(batch_id)
                file_name = os.listdir(self._config.base_path)[data_point_id]

                try:
                    data_point = self._load_function(file_name)
                except ValueError:
                    raise ValueError(file_name + " could not be loaded!")

                processed_data_point = [data_point]
                for fun in self._preprocess_pipeline():
                    processed_data_point = map(fun, processed_data_point)
                cropped_data_point = self._crop_input(list(processed_data_point[data_point_version]))
                batch.append(cropped_data_point)
            
            if len(batch) < self._config.batch_size:
                self._pad_batch(batch)
            
            self._current_batch = self._current_batch + 1

            yield np.stack(batch)

    def get_as_numpy_array(self):
        """Return the whole dataset in a numpy array
        This is used so that we can store images and histograms for Tensorboard
        """
        batch_iterator = self.batch_iter()
        this_epoch = self._current_epoch
        dataset = next(batch_iterator)[0]
        while this_epoch == self._current_epoch:
            dataset = np.concatenate((dataset, next(batch_iterator)[0]))
        return dataset, dataset

    def split(self, split_ratio):
        """Splits the Dataset into Test and Train 
        
        Args:
            split_ratio: Ratio of the data that should be in the train set.
            
        Returns:
            Two new Dataset objects
        """
        
        num_datapoints_train = int(split_ratio*self._config.num_datapoints)

        training_config = copy.deepcopy(self._config)
        training_config.num_datapoints = num_datapoints_train

        validation_config = copy.deepcopy(self._config)
        validation_config.num_datapoints = self._config.num_datapoints - num_datapoints_train

        training_indices = self._indices[:num_datapoints_train]
        validation_indices = self._indices[num_datapoints_train:]

        print("Number of datapoints in training set: " + str(len(training_indices)))
        print("Number of datapoints in test set: " + str(len(validation_indices)))

        training_set = type(self)()
        validation_set = type(self)()
        training_set.initialize(training_config, indices=training_indices)
        validation_set.initialize(validation_config, indices=validation_indices)

        return training_set, validation_set

    def get_size(self):
        """Returns the number of samples in the dataset.
        """
        
        return len(self._indices)

    def initialize(self, config, base_path=None, input_dimensions=None, batch_size=None, indices=None):
        """Construct Dataset

            Args:
                config: Instance of AbstractDatasetConfig
                base_path: Path where the datapoints are stored
                input_dimensions: Instance of utils.Dimension
                batch_size: Batch size
                indices: The indices of datapoints in this dataset

            Returns:
                Dataset
        """
        if base_path is not None:
            config.initialize(base_path, input_dimensions, batch_size)
        self._config = config

        if indices is None:
            self._indices = \
                list(range(self._config.augmentation_multiplicator * self._config.num_datapoints))
            rand.shuffle(self._indices)
        else:
            self._indices = indices
