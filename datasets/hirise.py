from datasets.dataset import Dataset
from planetaryimage.pds3image import PDS3Image
from datasets.dataset import DatasetConfig
import os
import numpy as np


class Hirise(Dataset):

    def _preprocess_pipeline(self):
        return [lambda x: (x / 127.5) - 1]

    def _load_function(self, file_name):
        file = os.path.join(self._config.base_path, file_name)
        pdsimage = PDS3Image.open(file)
        height, width = pdsimage.image.shape
        return np.reshape(pdsimage.image, (height, width, 1))

    def _crop_input(self, datapoint):
        height, width, depth = datapoint.shape
        crop_width = (width - self._config.input_dimensions.width) // 2
        crop_height = (height - self._config.input_dimensions.height) // 2
        if crop_width < 0 or crop_height < 0:
            raise ValueError("You chose input dimensions that are larger than the actual image.")
        if crop_height == 0:
            return datapoint[:, crop_width:-crop_width, :]
        if crop_width == 0:
            return datapoint[crop_height:-crop_height, :, :]
        return datapoint[crop_height:-crop_height, crop_width:-crop_width, :]


config = DatasetConfig(augmentation_multiplicator=1)
dataset = Hirise()

