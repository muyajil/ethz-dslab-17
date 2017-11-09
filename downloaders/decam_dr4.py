"""
This module downloads as much files as possible from HiRISE
"""

import argparse
import ftplib
from bs4 import BeautifulSoup
import urllib.request
import urllib.error
import os
import time


def download_images(download_location, min_images):
    """Download Decam images
    """
    ftp_server = "archive.noao.edu"
    base_path = "public/hlsp/ls/dr4/raw/"
    ftp = ftplib.FTP(ftp_server)
    ftp.connect()
    ftp.login()
    ftp.set_pasv(True)
    ftp.cwd(base_path)
    folders = ftp.nlst()
    images_downloaded = 0
    for folder in folders:
        ftp.cwd(folder)
        bricks = ftp.nlst()
        for brick in bricks:
            ftp.cwd(brick)
            files = ftp.nlst()
            for file in files:
                if 0 < min_images < images_downloaded:
                    return
                if "fits" in file:
                    file_path = os.path.join(download_location, file)
                    if os.path.exists(file_path):
                        continue
                    handle = open(os.path.join(download_location, file), 'wb')
                    ftp.retrbinary('RETR %s' % file, handle.write)
                    print("Downloaded image: {}".format(file))
                    images_downloaded += 1
            ftp.cwd("..")
        ftp.cwd("..")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("download_location", metavar="download_location", type=str, help="Path to download images to")
    parser.add_argument("--min_images", dest="min_images", type=int, help="Minimum number of images to download", default=-1)
    args = parser.parse_args()

    if not os.path.exists(args.download_location):
        os.makedirs(args.download_location)
    download_images(args.download_location, args.min_images)
