import httplib2
from bs4 import BeautifulSoup
import os
import urllib.request
import sys


class MarsRoverDownloader(object):

    _camera = None
    _data_path = None
    _already_downloaded = False

    def __init__(self, camera):
        self._camera = camera
        current_path = os.path.abspath(__file__)
        downloaders_path = os.path.abspath(os.path.join(current_path, os.pardir))
        root = os.path.abspath(os.path.join(downloaders_path, os.pardir))
        self._data_path = os.path.join(os.path.join(os.path.join(root, "data"), "mars_rover"), camera)
        if not os.path.exists(self._data_path):
            os.makedirs(self._data_path)
        else:
            self._already_downloaded = True

    def download(self, num_datapoints=sys.maxsize):
        if self._already_downloaded:
            return len(os.listdir(self._data_path)), self._data_path, self._camera, ".jpg"

        page = 0
        image_id = 0
        while image_id <= num_datapoints:
            page = page + 1
            url = "http://mars-ogler.com/?per-page=100&page=" + str(page) + "&cams=" + self._camera
            http = httplib2.Http()
            meta, response = http.request(url)

            document = BeautifulSoup(response, "html5lib")
            links = document.find_all("a", href=True)
            nasa_links = list(filter(lambda x: "http://mars.jpl.nasa.gov/msl-raw-images/" in x['href'], links))
            image_links = list(map(lambda x: x['href'], nasa_links[0::2]))

            if len(image_links) == 0:
                break
            else:
                for link in image_links:
                    image_id = image_id + 1
                    file_name = self._camera + "_" + str(image_id) + ".jpg"
                    urllib.request.urlretrieve(link, os.path.join(self._data_path, file_name))
                    if image_id % 100 == 0:
                        print("Downloaded " + str(image_id) + " images.")

        return image_id, self._data_path, self._camera, ".jpg"


if __name__ == '__main__':
    downloader = MarsRoverDownloader("mahli")
    print(downloader.download())
