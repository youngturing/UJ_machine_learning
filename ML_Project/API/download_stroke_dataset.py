import requests
import pickle
from zipfile import ZipFile

def download_data_zip(url: str, path: str) -> None:
    """
    :param url: address for downloading data .zip file
    :param path: path to saving file in project's 'data' directory
    """
    with open(path, 'wb') as file:
        req = requests.get(url)
        url_content = req.content
        pickle.dump(url_content, file)

def extract_data_to_csv(path: str, path_for_extracted_data: str) -> None:
    """
    :param path: path to saving file in project's 'data' directory
    :param path_for_extracted_data: path for data extracted to .csv file
    """
    with ZipFile(path, 'r') as zip_obj:
        zip_obj.extractall(path_for_extracted_data)

URL = 'https://storage.googleapis.com/kaggle-data-sets/1120859/1882037/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-' \
      'com@kaggle-161607.iam.gserviceaccount.com/20220124/auto/storage/goog4_request&X-Goog-Date=20220124T220940Z&X-Goog-Expires=259199&X-Goog-SignedHeaders=host&X-' \
      'Goog-Signature=6e7dbd2810026828a04fe1223925fd93399d50029450157bf8834713383a533f91d466863602bd6bc3765ed102680a0b60d2daf42262aa7eb67d3ccc184c0278eb43a426bc19e298c5bb94ea4f' \
      'c84f76b1c3a8daa1003448ac5e51a5ca46429bf93331c89ccabc8596bc65b16f4b709130c6fba73c2e4092c8049829c157a0e531658c650a45ab13438d0bf4b5f7718b162f328d7ec5686e0e0cefa7e2ca151a61a90' \
      '27eb0dd25720b218f75774de6990e920655f3f91739fdb6f9b6e493ce5f7f18a7bd9f87cd1fcbdddb6f0a1f54a3dd0ba3d4c4c6289a9637a0cc95fa80fde4a0cc980da26d1501c6e71d9b660ae8885e6ed5dd5d55b823c289addb834790'

PATH = '../API/data.zip'
PATH_FOR_EXTRACTED_DATA = '../API/data'

def main(path, path_for_extracted_data, url):
    download_data_zip(url, path)
    extract_data_to_csv(path, path_for_extracted_data)

if __name__ == '__main__':
    main(PATH, PATH_FOR_EXTRACTED_DATA, URL)