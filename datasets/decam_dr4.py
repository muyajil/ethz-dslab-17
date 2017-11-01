from datasets.dataset import Dataset
from astropy.io import fits
from datasets.dataset import DatasetConfig
import os
import numpy as np
import hashlib


class Decam(Dataset):

    def random_square(self, image):
        randints = np.random.random_integers(0, self._config.input_dimensions.width, 4)
        left = min(randints[:2])
        right = max(randints[:2])
        top = max(randints[2:])
        bottom = min(randints[2:])
        for i in range(left, right):
            for j in range(bottom, top):
                image[i, j, 0] = 1
        return image

    def _preprocess_pipeline(self):
        return []

    def set_seed(self, file_name):
        hash_string = hashlib.md5(file_name.encode()).hexdigest()
        ints = [int(s) for s in list(hash_string) if s.isdigit()]
        np.random.seed(sum(ints))

    def _load_function(self, file_name):
        self.set_seed(file_name)
        file = os.path.join(self._config.base_path, file_name)
        try:
            image_data = fits.getdata(file, ignore_missing_end=True)
        except OSError:
            raise OSError("Image cannot be opened: {}".format(file_name))
        height, width = image_data.shape
        normalized_image = (np.log(image_data) / 15) - 1
        # plt.imshow(pdsimage.image, cmap='gray')
        return np.reshape(normalized_image, (height, width, 1))

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
dataset = Decam()

