from datasets.dataset import Dataset
from downloaders.mars_rover import MarsRoverDownloader
import os
from scipy import misc


class MarsRoverMahli(Dataset):

    def _download_data(self):
        downloader = MarsRoverDownloader("mahli")
        return downloader.download()

    def _load_function(self, file_id):
        file_path = os.path.join(self._base_path, self._base_name + "_" + file_id + self._file_ending)
        return misc.imread(file_path) # returns a tensor of shape length x width x 3 for rgb

    def _preprocess_pipeline(self):
        return []


dataset = MarsRoverMahli(augmentation_multiplicator=1)
