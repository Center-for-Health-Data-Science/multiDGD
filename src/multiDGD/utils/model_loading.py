import os
import requests, zipfile

BASE_URL = 'https://api.figshare.com/v2'

def get_figshare_file(f_url, f_name, f_zip=False):
    # check if file already exists
    if not os.path.exists(f_name):
        file_url = BASE_URL + f_url

        file_response = requests.get(file_url).json()
        file_download_url = file_response['download_url']
        response = requests.get(file_download_url, stream=True)
        with open(f_name, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        if f_zip:
            with zipfile.ZipFile(f_name, 'r') as zip_ref:
                zip_ref.extractall('.')
    else:
        print(f'File {f_name} already exists')