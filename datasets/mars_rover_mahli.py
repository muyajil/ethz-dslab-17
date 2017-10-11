from datasets.dataset import Dataset
from downloaders.mars_rover import MarsRoverDownloader
import os
from scipy.ndimage import imread


class MarsRoverMahli(Dataset):

    def _download_data(self):
        downloader = MarsRoverDownloader("mahli")
        return downloader.download()

    def _load_function(self, file_id, image_dim):
        file_path = os.path.join(self._base_path, self._base_name + "_" + str(file_id) + self._file_ending)
        image = imread(file_path) # returns a tensor of shape length x width x 3 for rgb
        return self._crop_image(image, image_dim)

    def _preprocess_pipeline(self):
        return []


dataset = MarsRoverMahli(augmentation_multiplicator=1)
