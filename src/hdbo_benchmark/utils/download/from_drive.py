"""
Taken and adapted from:
https://stackoverflow.com/a/39225272
"""

import hashlib

import requests


def download_file_from_google_drive(file_id, destination, md5_checksum: str = None):
    URL = "https://docs.google.com/uc?export=download&confirm=1"

    session = requests.Session()

    response = session.get(URL, params={"id": file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {"id": file_id, "confirm": token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)

    if md5_checksum is not None:
        with open(destination, "rb") as f:
            assert hashlib.md5(f.read()).hexdigest() == md5_checksum


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
