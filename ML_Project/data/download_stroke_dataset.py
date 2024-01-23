import subprocess

from zipfile import ZipFile


def download_data_zip() -> None:
    subprocess.run([
        'kaggle', 'datasets', 'download', '-d', 'fedesoriano/stroke-prediction-dataset'
    ])


def extract_data_to_csv(path: str) -> None:
    """
    Parameters
    ----------
        path: downloaded filename with healthcare data.
    """
    print(path)
    with ZipFile(path, 'r') as zip_obj:
        zip_obj.extractall()


def main(path):
    download_data_zip()
    extract_data_to_csv(path)


if __name__ == '__main__':
    path = './stroke-prediction-dataset.zip'
    main(path)
