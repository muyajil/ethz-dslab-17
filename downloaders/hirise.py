"""
This module downloads as much files as possible from HiRISE
"""

import argparse
import httplib2
from bs4 import BeautifulSoup
import urllib.request
import urllib.error
import os
import time


def gather_links(download_location, num_images):
    """Download HiRISE images
    """
    base_path = "https://hirise-pds.lpl.arizona.edu/PDS/EDR/ESP/"
    http_client = httplib2.Http()
    meta, response = http_client.request(base_path)
    level_0 = BeautifulSoup(response, "html5lib")
    level_0_links = list(map(lambda x: x['href'], level_0.find_all("a", href=True)))[1:]
    image_links = []
    downloaded_images = 0
    for level_0_link in level_0_links:
        if len(image_links) > 100:
            download_images(image_links, download_location, downloaded_images)
            downloaded_images = downloaded_images + len(image_links)
            image_links = []
        if downloaded_images > num_images:
            return
        meta, response = http_client.request(base_path + level_0_link)
        level_1 = BeautifulSoup(response, "html5lib")
        level_1_links = list(map(lambda x: x['href'], level_1.find_all("a", href=True)))[1:]
        for level_1_link in level_1_links:
            meta, response = http_client.request(base_path + level_0_link + level_1_link)
            level_2 = BeautifulSoup(response, "html5lib")
            level_2_links = list(map(lambda x: x['href'], level_2.find_all("a", href=True)))[1:]
            image_links.extend(list(map(lambda x: base_path + level_0_link + level_1_link + x, level_2_links)))


def download_images(image_links, download_location, image_id):
    print(str(len(image_links)) + " Images to be downloaded.")
    print("Starting download...")
    for image_link in image_links:
        file_name = "image_" + str(image_id) + ".IMG"
        image_id = image_id + 1
        try:
            urllib.request.urlretrieve(image_link, os.path.join(download_location, file_name))
        except urllib.error.HTTPError:
            time.sleep(180)
            image_id = image_id - 1
        if image_id % 100 == 0:
            print("Downloaded " + str(image_id) + " images.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("download_location", metavar="download_location", type=str, help="Path to download images to")
    parser.add_argument("--num_images", dest="num_images", type=int, help="Number of images to download", default=-1)

    args = parser.parse_args()

    gather_links(args.download_location, args.num_images)
