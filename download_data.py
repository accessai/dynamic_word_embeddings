import os
from bs4 import BeautifulSoup
import urllib.request
import subprocess

from tqdm import tqdm


def scrap_download_links(url):
    resp = urllib.request.urlopen(url)
    soup = BeautifulSoup(resp, from_encoding=resp.info().get_param('charset'))

    links = []
    for link in soup.find_all('a', href=True):
        url = link['href']

        if '5gram' in url and 'eng' in url:
            links.append(url)

    print(f"Total link counts {len(links)}")

    return links


def download_files(links, output_dir):
    os.makedirs(output_dir,exist_ok=True)
    for url in tqdm(links, 'Downloading...'):
        filename = url.split("/")[-1]
        file_path = os.path.join(output_dir, filename)
        # subprocess.run(["wget", "-O", file_path, url])
        subprocess.run(["curl", url, "--output", file_path])

if __name__ == '__main__':

    url = "http://storage.googleapis.com/books/ngrams/books/datasetsv2.html"
    links = scrap_download_links(url)[21:22]
    download_files(links, output_dir='./data')