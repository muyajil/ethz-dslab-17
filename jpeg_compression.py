import os
import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt
from PIL import Image, ImageFile
from pylab import rcParams
import random
rcParams['figure.figsize'] = 20, 5


BASE_PATH = "./data/decam_dr4"
NUM_IMAGES = 50


def get_image_iterator():
    file_names = os.listdir(BASE_PATH)
    while True:
        file_name = random.choice(file_names)
        file = os.path.join(BASE_PATH, file_name)
        image_data = fits.getdata(file, ignore_missing_end=True)
        # plt.imshow(pdsimage.image, cmap='gray')
        yield image_data, os.path.getsize(file)


def log10(x):
    numerator = np.log(x)
    denominator = np.log(10)
    return np.divide(numerator, denominator)


def compute_psnr(original, reconstructed, peak=255.0):
    rescaled_image = (original / (2 ** 16 / 255)).astype('uint8').squeeze()
    mse = np.mean(np.multiply(rescaled_image - reconstructed, rescaled_image - reconstructed))
    rmse = np.sqrt(mse)
    return 20 * log10(np.divide(peak, rmse))


def crop_input(datapoint):
    height, width, depth = datapoint.shape
    crop_width = (width - 256) // 2
    crop_height = (height - 256) // 2
    if crop_width < 0 or crop_height < 0:
        raise ValueError("You chose input dimensions that are larger than the actual image.")
    if crop_height == 0:
        return datapoint[:, crop_width:-crop_width, :]
    if crop_width == 0:
        return datapoint[crop_height:-crop_height, :, :]
    return datapoint[crop_height:-crop_height, crop_width:-crop_width, :]


def get_jpeg_encoded_image(raw_image, quality):
    ImageFile.MAXBLOCK = 2**20
    rescaled_image = (raw_image/(2**16/255)).astype('uint8').squeeze()
    img = Image.fromarray(rescaled_image, 'L')
    img.save("out.jpg", "JPEG", quality=quality, optimize=True, progressive=True)
    img = Image.open("out.jpg")
    os.remove("out.jpg")
    return np.array(img), os.path.getsize("out_{}.jpg".format(quality))


if __name__ == '__main__':
    images = get_image_iterator()
    psnr_and_compression = []
    for i, (image, raw_size) in enumerate(images):
        if i >= NUM_IMAGES:
            break
        for quality in range(1, 101, 1):
            jpeg, jpeg_size = get_jpeg_encoded_image(image, quality)
            psnr = compute_psnr(image, jpeg)
            compression_rate = raw_size/jpeg_size
            psnr_and_compression.append((compression_rate, psnr))

    sorted_p_c = sorted(psnr_and_compression, key=lambda x: x[0])
    max_compression_rate = sorted_p_c[-1][0]
    min_compression_rate = sorted_p_c[0][0]
    indices = list(range(int(min_compression_rate)+1, int(max_compression_rate)+1, 10))
    means = []
    final_indices = indices
    for i in indices:
        current_psnrs = []
        while sorted_p_c[0][0] <= i:
            current_psnrs.append(sorted_p_c.pop(0)[1])
        if current_psnrs:
            means.append(np.median(current_psnrs))
        else:
            means.append(0)

    xarr = []
    yarr = []

    for mean, index in zip(means, indices):
        if mean != 0:
            yarr.append(mean)
            xarr.append(index)

    plt.plot(xarr, yarr, 'b-o', lw=1.0)
    plt.ylabel("PSNR")
    plt.xlabel("Compression Rate")
    plt.title("Median PSNR per compression rate")
    plt.savefig("psnr_jpeg.png")
