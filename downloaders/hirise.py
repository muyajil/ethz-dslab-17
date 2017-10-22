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


def gather_links(download_location, min_images):
    """Download HiRISE images
    """
    base_path = "https://hirise-pds.lpl.arizona.edu/PDS/EDR/ESP/"
    http_client = httplib2.Http()
    meta, response = http_client.request(base_path)
    level_0 = BeautifulSoup(response, "html5lib")
    level_0_links = list(map(lambda x: x['href'], level_0.find_all("a", href=True)))[1:]
    image_links = []
    for level_0_link in level_0_links[4:]: # TODO remove [1:] this is just temporary
        if len(image_links) > 100:
            download_images(image_links, download_location)
            image_links = []
        if len(os.listdir(download_location)) > min_images:
            return
        meta, response = http_client.request(base_path + level_0_link)
        level_1 = BeautifulSoup(response, "html5lib")
        level_1_links = list(map(lambda x: x['href'], level_1.find_all("a", href=True)))[1:]
        for level_1_link in level_1_links:
            meta, response = http_client.request(base_path + level_0_link + level_1_link)
            level_2 = BeautifulSoup(response, "html5lib")
            level_2_links = list(map(lambda x: x['href'], level_2.find_all("a", href=True)))[1:]
            image_links.extend(list(map(lambda x: base_path + level_0_link + level_1_link + x, level_2_links)))


def download_images(image_links, download_location):
    print(str(len(image_links)) + " Images to be downloaded.")
    print("Starting download...")
    for image_link in image_links:
        file_name = image_link.split("/")[-1]
        file_path = os.path.join(download_location, file_name)
        if os.path.exists(file_path):
            continue
        try:
            urllib.request.urlretrieve(image_link, file_path)
        except urllib.error.HTTPError:
            time.sleep(180)
            if os.path.exists(file_path):
                os.remove(file_path)
                print("HTTP Error occured")
                continue
        except urllib.error.ContentTooShortError:
            print("Image download error: " + file_name)
            os.remove(file_path)
            continue
        if os.path.getsize(file_path) < 15000000: # Some images are not big enough, minimum file size is 15mb
            os.remove(file_path)
            print("Image was too small: " + file_name)
            continue
        print("Downloaded image: " + file_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("download_location", metavar="download_location", type=str, help="Path to download images to")
    parser.add_argument("--min_images", dest="min_images", type=int, help="Minimum number of images to download", default=-1)
    args = parser.parse_args()

    if not os.path.exists(args.download_location):
        os.makedirs(args.download_location)
    gather_links(args.download_location, args.min_images)
