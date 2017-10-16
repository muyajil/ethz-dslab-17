from datasets.abstractdataset import AbstractDataset
from planetaryimage.pds3image import PDS3Image
from datasets.abstractdataset import DatasetConfig
import os


class Hirise(AbstractDataset):

    def _preprocess_pipeline(self):
        return []

    def _load_function(self, file_id):
        file = os.path.join(self._config.base_path, self._config.base_name + str(file_id) + self._config.file_ending)
        pdsimage = PDS3Image.open(file)
        return pdsimage.image

    def _crop_input(self, datapoint):
        height, width = datapoint.shape
        crop_width = (width - self._config.input_dimensions.width) // 2
        crop_height = (height - self._config.input_dimensions.height) // 2
        if crop_width < 0 or crop_height < 0:
            raise ValueError("You chose input dimensions that are larger than the actual image.")
        if crop_height == 0:
            return datapoint[:, crop_width:-crop_width]
        if crop_width == 0:
            return datapoint[crop_height:-crop_height, :]
        return datapoint[crop_height:-crop_height, crop_width:-crop_width]


config = DatasetConfig(augmentation_multiplicator=1, base_name='image_', file_ending='.IMG')
dataset = Hirise()

