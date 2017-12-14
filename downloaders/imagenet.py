import argparse
import os
import urllib.request
import urllib.error


def download_images(download_location, min_images):
    with open('fall11_urls.txt', encoding='latin-1') as file:
        downloaded_images = 0
        for line in file.readlines():
            if downloaded_images > min_images:
                break
            link = line.split()[1]
            path_to_image = os.path.join(download_location, "image_{}.jpg".format(downloaded_images))
            try:
                urllib.request.urlretrieve(link, path_to_image)
                if os.path.getsize(path_to_image) > 10000:
                    print("Downloaded: image_{}.jpg".format(downloaded_images))
                    downloaded_images += 1
                else:
                    print("Link: {} is corrupt".format(link))
                    os.remove(path_to_image)
            except (urllib.error.HTTPError, urllib.error.URLError):
                print("Link: {} is broken".format(link))
                continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("download_location", metavar="download_location", type=str, help="Path to download images to")
    parser.add_argument("--min_images", dest="min_images", type=int, help="Minimum number of images to download",
                        default=-1)
    args = parser.parse_args()

    if os.path.exists(args.download_location):
        os.makedirs("{}_new".format(args.download_location))
        download_images(args.download_location, args.min_images)
    else:
        os.makedirs(args.download_location)
        download_images(args.download_location, args.min_images)
