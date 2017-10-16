from datasets.abstractdataset import AbstractDataset
from planetaryimage.pds3image import PDS3Image


class Hirise(AbstractDataset):

    def _preprocess_pipeline(self):
        return []

    def _load_function(self, file_id):
        filename = self._config.base_path + self._config.base_name + str(file_id) + self._config.file_ending
        pdsimage = PDS3Image.open(filename)
        return pdsimage.image


dataset = Hirise()

