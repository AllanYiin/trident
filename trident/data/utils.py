from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools
import base64
import datetime
import hashlib
import json
import time
import os
import io
import random
import platform
import shutil
import sys
import tarfile
import urllib.request
from urllib.parse import urlencode
import zipfile
from bs4 import BeautifulSoup
import six
from scipy.io import loadmat
from tqdm import tqdm

from trident.backend.common import *
from trident.context import make_dir_if_need, split_path, sanitize_path

try:
    from urllib.request import urlretrieve
except ImportError:
    from six.moves.urllib.request import urlretrieve

__all__: object = ['is_connected', 'ensure_dir', 'ensure_parent_dir', 'TqdmProgress', 'calculate_md5',
                   'check_integrity', 'download_file', 'get_onedrive_directdownload',
                   'download_file_from_google_drive', 'download_file_from_onedrive', 'get_image_from_google_drive',
                   'get_file_from_google_drive',
                   'download_model_from_google_drive', 'download_model_from_onedrive', 'extract_archive', 'pickle_it',
                   'unpickle', 'save_dict_as_h5',
                   'read_dict_from_h5', 'get_file_create_time', 'read_mat']


def prepare_chrome_options():
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('blink-settings=imagesEnabled=false')
    chrome_options.add_argument('--disable-logging')
    chrome_options.add_argument(f"--window-size=1920,1440")
    chrome_options.add_argument('--hide-scrollbars')
    chrome_options.add_argument('--disable-blink-features=AutomationControlled')
    chrome_options.add_argument("--proxy-server='direct://'")
    chrome_options.add_argument("--proxy-bypass-list=*")
    chrome_options.add_argument('--ignore-certificate-errors')
    chrome_options.add_argument("--password-store=basic")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--enable-automation")
    chrome_options.add_argument("--disable-browser-side-navigation")
    chrome_options.add_argument("--disable-web-security")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-infobars")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--disable-setuid-sandbox")
    chrome_options.add_argument("--disable-software-rasterizer")
    return chrome_options


user_agents = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.2151.97',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.2151.97',
    'Mozilla/5.0 (Linux; Android 10; HD1913) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.6045.193 Mobile Safari/537.36 EdgA/119.0.2151.78',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.131 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:88.0) Gecko/20100101 Firefox/88.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.131 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Version/14.1.2 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Version/14.1 Safari/537.36',
    'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:89.0) Gecko/20100101 Firefox/89.0',
    'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:88.0) Gecko/20100101 Firefox/88.0'
]


def is_connected():
    try:
        urllib.request.urlopen('https://google.com')  # Python 3.x
        return True
    except:
        return False


def ensure_dir(dirpath):
    if not os.path.exists(dirpath): os.makedirs(dirpath)


def ensure_parent_dir(childpath):
    ensure_dir(os.path.dirname(childpath))


class TqdmProgress(tqdm):
    """Provides `update_to(n)` which uses `tqdm.update(delta_n)`."""

    def update_to(self, b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        pbar.update(progress - pbar.n)  # will also set self.n = b * bsize


def calculate_md5(fpath, chunk_size=1024 * 1024):
    """Calculate whether the file 's md5 hash is the same as given md5 .

    Args:
        fpath ():
        chunk_size ():

    Returns:

    """
    md5 = hashlib.md5()
    with open(fpath, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            md5.update(chunk)
    return md5.hexdigest()


def check_integrity(fpath, md5=None):
    if not os.path.isfile(fpath):
        return False
    if md5 is None:
        return True
    fmd5 = calculate_md5(fpath)
    return md5 == fmd5


def _write_h(dirname, is_downloaded=False, is_extracted=False, tag=None):
    _h_path = os.path.join(dirname, 'status.json')
    _h = {}
    if os.path.exists(_h_path):
        try:
            with open(_h_path) as f:
                _h = json.load(f)
        except ValueError:
            _h = {}
    if tag is None:
        _h['is_downloaded'] = is_downloaded
        _h['is_extracted'] = is_extracted
    else:
        if tag not in _h:
            _h[tag] = {}
        _h[tag]['is_downloaded'] = is_downloaded
        _h[tag]['is_extracted'] = is_extracted

    try:
        with open(_h_path, 'w') as f:
            f.write(json.dumps(_h, indent=4))
    except IOError:
        # Except permission denied.
        pass


def _read_h(dirname):
    _h = {}
    _h_path = os.path.join(dirname, 'status.json')
    if os.path.exists(_h_path):
        try:
            with open(_h_path) as f:
                _h = json.load(f)
        except ValueError:
            _h = {}
    return _h


def _delete_h(dirname):
    _h_path = os.path.join(dirname, 'status.json')
    if os.path.exists(_h_path):
        os.remove(_h_path)


def download_file(src, dirname, filename, desc=''):
    _h = _read_h(dirname)

    is_downloaded = _h[filename].get('is_downloaded', False) if filename in _h else _h.get('is_downloaded', False)

    if os.path.exists(os.path.join(dirname, filename)) and _h != {} and is_downloaded == True:
        print('archive file is already existing, donnot need download again.')
        return True
    else:
        if os.path.exists(os.path.join(dirname, filename)):
            os.remove(os.path.join(dirname, filename))

        try:
            with TqdmProgress(unit='B', unit_scale=True, leave=True, miniters=10,
                              desc=desc) as t:  # all optional kwargs
                urlretrieve(src, filename=os.path.join(dirname, filename), reporthook=t.update_to, data=None)
                _write_h(dirname, True, False, tag=filename)
            return True
        except Exception as e:
            _write_h(dirname, False, False, tag=filename)
            print('***Cannot download data,.\n', flush=True)
            print(e)
            return False


def get_onedrive_directdownload(onedrive_link):
    """

    Args:
        onedrive_link ():

    Returns:

    Examples:
        >>> link='https://1drv.ms/u/s!AsqOV38qroofiZrqNAQvo2CuX_cyWQE?e=JW28uv'
        >>> new_link=get_onedrive_directdownload(link)
        >>> print(new_link)
        'https://1drv.ms/u/s!AsqOV38qroofiZrqNAQvo2CuX_cyWQE?e=JW28uv'

    """
    data_bytes64 = base64.b64encode(bytes(onedrive_link, 'utf-8'))
    data_bytes64_String = data_bytes64.decode('utf-8').replace('/', '_').replace('+', '-').rstrip("=")
    resultUrl = f"https://api.onedrive.com/v1.0/shares/u!{data_bytes64_String}/root/content"
    return resultUrl


def download_file_from_google_drive(file_id, dirname=None, filename=None, md5=None, need_up_to_date=False):
    """Download a Google Drive file from  and place it in root.

    Args:
        need_up_to_date (bool): If True, trident will re-download this file every-times.
        file_id (str): id of file to be downloaded
        dirname (str): Directory to place downloaded file in
        filename (str, optional): Name to save the file under. If None, use the id of the file.
        md5 (str, optional): MD5 checksum of the download. If None, do not check

    Examples:
        >>> download_file_from_google_drive('mbo_bisenetV10_model.pth.tar','1xmSGwOZOo_rKGlaSIhCzaqx5VTw7cyrb')

    """
    # Based on https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url
    import requests

    url = 'https://drive.usercontent.google.com/uc?'
    params = {'id': file_id, 'authuser': '0', 'export': 'download'}
    dirname = os.path.join(get_trident_dir(), 'downloads') if not dirname else dirname
    filename = file_id if not filename else filename

    fpath = os.path.join(dirname, filename)
    _h = _read_h(dirname)

    make_dir_if_need(fpath)
    is_downloaded = _h[filename].get('is_downloaded', False) if filename in _h else _h.get('is_downloaded', False)

    if os.path.exists(fpath) and os.path.isfile(
            fpath) and _h != {} and is_downloaded == True and need_up_to_date == False:
        print('archive file is already existing, donnot need download again.')
        return True
    else:
        try:
            session = requests.Session()
            options_headers = {
                "Origin": "https://drive.google.com",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "x-drive-first-party,x-json-requested",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36",
            }

            options_response = session.options(url, headers=options_headers)
            if options_response.status_code == 200:
                post_headers = {
                    "Origin": "https://drive.google.com",
                    "Referer": "https://drive.google.com/",
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36",
                    "X-Drive-First-Party": "DriveWebUi",
                    "X-JSON-Requested": "true",
                    "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8",
                    "Accept": "*/*",
                    "Accept-Encoding": "gzip, deflate, br, zstd",
                }
                post_response = session.post(url,  params=params, headers=post_headers)

            # token = _get_confirm_token(response)
            # if token:
            #     response2 = session.get(url, params=params, stream=True)
            # else:
            jresult=json.loads(post_response.text.split('\n')[-1])
            download_url=jresult['downloadUrl']
            headers = {
                "User-Agent": "Mozilla/5.0",
                "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
                "accept-encoding": "gzip, deflate, br, zstd",
            }

            response2 = session.get(download_url, headers=headers,
                                     stream=True)
            file_size = int(response2.headers.get('Content-Length', 0))
            _save_response_content(response2, fpath)
            if md5:
                is_md5_valid = check_integrity(fpath, md5)
                if is_md5_valid:
                    return fpath
                else:
                    return None
            return fpath
        except Exception as e:
            PrintException()
            _write_h(dirname, False, False, filename)
            print('***Cannot download data, so the data provider cannot initialized.\n', flush=True)
            print(
                '***Please check your internet or download  files from following url in another computer, \n and then put them into {0}\n {1} '.format(
                    dirname,
                    'https://drive.google.com/open?id={0}'.format(file_id)), flush=True)
            print(e)
            return None


def download_file_from_onedrive(onedrive_path, dirname, filename=None, md5=None):
    """Download a OneDrive file from  and place it in root.

    Args:
        onedrive_path (str): id of file to be downloaded
        dirname (str): Directory to place downloaded file in
        filename (str, optional): Name to save the file under. If None, use the id of the file.
        md5 (str, optional): MD5 checksum of the download. If None, do not check


    Examples:
    >>> link='https://1drv.ms/u/s!AsqOV38qroofiZrqNAQvo2CuX_cyWQE?e=JW28uv'
    >>> new_link=get_onedrive_directdownload(link)
    >>> download_file_from_onedrive(link,'~/.trident/models','models_md5.json')
    archive file is already existing, donnot need download again.
    True
    >>> print(new_link)
    https://api.onedrive.com/v1.0/shares/u!aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBc3FPVjM4cXJvb2ZpWnJxTkFRdm8yQ3VYX2N5V1FFP2U9SlcyOHV2/root/content

    """
    # Based on https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url
    url = get_onedrive_directdownload(onedrive_path)
    # if os.path.exists(os.path.join(dirname, filename)):
    #     print('archive file is already existing, donnot need download again.')
    dest_path = os.path.join(os.path.join(get_trident_dir(), 'models'), filename)
    if os.path.exists(dest_path) and os.path.isfile(dest_path) and (
            datetime.datetime.now() - get_file_modified_time(dest_path)).seconds < 12 * 60 * 60:
        print('archive file is already existing, donnot need download again.')
        return True
    else:
        if os.path.exists(dest_path):
            os.remove(dest_path)

        try:
            with TqdmProgress(unit='B', unit_scale=True, leave=True, miniters=10, desc='') as t:  # all optional kwargs
                urlretrieve(url, filename=dest_path, reporthook=t.update_to, data=None)
                _write_h(dirname, True, False, filename)
            return True
        except Exception as e:
            _write_h(dirname, False, False, filename)
            print('***Cannot download data, so the data provider cannot initialized.\n', flush=True)
            print(
                '***Please check your internet or download  files from following url in another computer, \n and then put them into {0} '.format(
                    dirname), flush=True)

            print(e)
            return False


def get_image_from_google_drive(file_id):
    """Download a Google Drive image  and place it in root.

    Args:
        file_id (str): id of file to be downloaded


    Returns:
        the file path of this downloaded image

    Examples:
        >>> detectorp=get_image_from_google_drive('1DO0iCg5Pmyk_F8s-ICgiy3OGcG_hbP6e')


    """

    import requests
    url = 'https://drive.google.com/uc?export=download'

    filename = file_id

    _session = get_session()
    _trident_dir = _session.trident_dir
    dirname = os.path.join(_trident_dir, 'download')
    make_dir_if_need(dirname)
    fpath = os.path.join(dirname, filename)

    if os.path.exists(fpath):
        os.remove(fpath)
    try:
        session = requests.Session()
        response = session.get(url, params={'id': file_id}, stream=True)
        content_type = response.headers.get('content-type')
        filename = file_id + '.' + content_type.split('/')[-1]
        fpath = os.path.join(dirname, filename)
        with open(fpath, 'wb') as f:
            f.write(response.content)
        # token = _get_confirm_token(response)
        # if token:
        #     params = {'id': file_id, 'confirm': token}
        #     response = session.get(url, params=params, stream=True)
        # _save_response_content(response, fpath)
        return fpath
    except Exception as e:
        print('***Cannot download data, so the data provider cannot initialized.\n', flush=True)
        print(
            '***Please check your internet or download  files from following url in another computer, \n and then put them into {0}\n {1} '.format(
                dirname,
                'https://drive.google.com/open?id={0}'.format(
                    file_id)), flush=True)
        print(e)
        return None


def get_file_from_google_drive(filename, file_id):
    """Download a Google Drive image  and place it in root.

    Args:
        filename (str, optional): Name to save the file under. If None, use the id of the file.
        file_id (str): id of file to be downloaded

    Returns:
        the file path of this downloaded image

    Examples:

        >>> get_file_from_google_drive('efficientnet-b0.pth','1bxnoDerzoNfiZZLft4ocD3DAgx4v6aTN')

    """

    return download_file_from_google_drive(file_id, dirname=os.path.join(get_trident_dir(), 'download'),
                                           filename=filename, md5=None, need_up_to_date=True)


def download_model_from_google_drive(file_id, dirname, filename=None, md5=None):
    """Download a Google Drive file from  and place it in root.

    Args:
        file_id (str): id of file to be downloaded
        dirname (str): Directory to place downloaded file in
        filename (str, optional): Name to save the file under. If None, use the id of the file.
        md5 (str, optional): MD5 checksum of the download. If None, do not check
    """
    # Based on https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url
    need_download_model_json = True
    need_download = True
    filename = file_id if not filename else filename
    dirname = os.path.join(get_trident_dir(), 'models')

    make_dir_if_need(dirname)
    fpath = os.path.join(dirname, filename)
    if os.path.exists(os.path.join(dirname, 'models_md5.json')) and os.path.isfile(
            os.path.join(dirname, 'models_md5.json')) and (
            datetime.datetime.now() - get_file_modified_time(
        os.path.join(dirname, 'models_md5.json'))).seconds < 24 * 60 * 60:
        need_download_model_json = False

    download_file_from_google_drive(file_id='12XLjt9Zcaoo90WGG6R5N0U6Sf_KBZZn_',
                                    dirname=os.path.join(get_trident_dir(), 'models'), filename='models_md5.json',
                                    md5=None, need_up_to_date=need_download_model_json)
    if os.path.exists(os.path.join(dirname, 'models_md5.json')):
        with open(os.path.join(dirname, 'models_md5.json')) as f:
            models_md5 = json.load(f)

    if os.path.exists(fpath):
        if check_integrity(fpath, models_md5[filename]):
            need_download = False
            print('model file is already existing, donnot need download again.')
        else:
            print('Your pretrained model has newer version, will you want to update it?')
            ans = input('(Y/N) << ').lower()
            if ans in ['yes', 'y']:
                os.remove(fpath)
            else:
                need_download = False

    if need_download:
        return download_file_from_google_drive(file_id, dirname=dirname, filename=filename,
                                               md5=models_md5[filename], need_up_to_date=need_download)
    else:
        return fpath


# https://1drv.ms/u/s!AsqOV38qroofiZrqNAQvo2CuX_cyWQE?e=Aa8v7D

def download_model_from_onedrive(onedrive_path, dirname, filename=None, md5=None):
    """Download a Google Drive file from  and place it in root.

    Args:
        onedrive_path (str): id of file to be downloaded
        dirname (str): Directory to place downloaded file in
        filename (str, optional): Name to save the file under. If None, use the id of the file.
        md5 (str, optional): MD5 checksum of the download. If None, do not check
    """
    # Based on https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url
    url = get_onedrive_directdownload(onedrive_path)

    fpath = os.path.join(dirname, filename)
    isload = False
    models_md5 = None
    need_download = True
    check_internet = None
    try:
        if os.path.exists(os.path.join(dirname, 'models_md5.json')) and os.path.isfile(
                os.path.join(dirname, 'models_md5.json')) and (
                datetime.datetime.now() - get_file_modified_time(
            os.path.join(dirname, 'models_md5.json'))).seconds < 24 * 60 * 60:
            with open(os.path.join(dirname, 'models_md5.json')) as f:
                models_md5 = json.load(f)
        else:
            download_file_from_onedrive("https://1drv.ms/u/s!AsqOV38qroofiZrqNAQvo2CuX_cyWQE?e=Aa8v7D", dirname,
                                        'models_md5.json')

            check_internet = True
            if os.path.exists(os.path.join(dirname, 'models_md5.json')):
                with open(os.path.join(dirname, 'models_md5.json')) as f:
                    models_md5 = json.load(f)
    except Exception as e:
        print(e)
        PrintException()
        check_internet = False

    if check_internet == False:
        if os.path.exists(os.path.join(dirname, filename)):
            print('internet connect  error,model file is already existing, donnot need download again.')
            return True
        else:
            _write_h(dirname, False, False, filename)
            print('***Cannot download data, so the data provider cannot initialized.\n', flush=True)
            print(
                '***Please check your internet or download  files from following url in another computer, \n and then '
                'put them into {0}\n {1} '.format(dirname, url), flush=True)

        return False
    else:

        try:
            if os.path.exists(os.path.join(dirname, filename)):
                if check_integrity(os.path.join(dirname, filename), models_md5[filename]):
                    need_download = False
                    print('model file is already existing, donnot need download again.')
                else:
                    print('Your pretrained model has newer version, will you want to update it?')
                    ans = input('(Y/N) << ').lower()
                    if ans in ['yes', 'y']:
                        os.remove(os.path.join(dirname, filename))
                    else:
                        need_download = False

            if need_download:
                with TqdmProgress(unit='B', unit_scale=True, leave=True, miniters=10,
                                  desc='') as t:  # all optional kwargs
                    urlretrieve(url, filename=os.path.join(dirname, filename), reporthook=t.update_to, data=None)
                _write_h(dirname, True, False, filename)
                if check_integrity(os.path.join(dirname, filename), models_md5[filename]):
                    print('model file is downloaded and validated.')
                else:
                    print('model file is downloaded but not match md5.')
        except Exception as e:
            _write_h(dirname, False, False, filename)
            print(e)
            print('***Cannot download data, so the data provider cannot initialized.\n', flush=True)
            print(
                '***Please check your internet or download  files from following url in another computer, \n and then '
                'put them into {0}\n {1} '.format(dirname, url), flush=True)
            print(e)

        return False


def _get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def _save_response_content(response, destination, chunk_size=32768):
    folder, file = os.path.split(destination)
    file_size = int(response.headers.get('Content-Length', 0))
    progress = 0
    with tqdm(total=file_size, unit="B", unit_scale=True) as progress_bar:
        with open(destination, "wb") as file:
            for data in response.iter_content(chunk_size):
                progress_bar.update(len(data))
                file.write(data)


def extract_archive(file_path, target_folder=None, archive_format='auto'):
    """Extracts an archive if it matches tar, tar.gz, tar.bz, or zip formats.

    Args:
        file_path: path to the archive file
        target_folder: path to extract the archive file
        archive_format: Archive format to try for extracting the file.
            Options are 'auto', 'tar', 'zip', and None.
            'tar' includes tar, tar.gz, and tar.bz files.
            The default 'auto' is ['tar', 'zip'].
            None or an empty list will return no matches found.

    Returns:
        True if a match was found and an archive extraction was completed,
        False otherwise.
    """
    folder, file, ext = split_path(file_path)
    filename = file + ext
    try:

        _h = _read_h(target_folder)
        is_extracted = _h[filename].get('is_extracted', False) if filename in _h else _h.get('is_extracted', False)
        if _h != {} and is_extracted == True and os.path.exists(file_path):
            print('extraction is finished, donnot need extract again.')
            return True
        if archive_format is None:
            return False
        if archive_format == 'auto':
            archive_format = ['tar', 'zip']
        if isinstance(archive_format, six.string_types):
            archive_format = [archive_format]

        is_match_fn = tarfile.is_tarfile
        open_fn = tarfile.open
        for archive_type in archive_format:
            if archive_type == 'tar':
                open_fn = tarfile.open
                is_match_fn = tarfile.is_tarfile
            if archive_type == 'zip':
                open_fn = zipfile.ZipFile
                is_match_fn = zipfile.is_zipfile

            if is_match_fn(file_path):
                print('Starting to decompress the archive....')
                with open_fn(file_path) as archive:
                    try:
                        archive.extractall(target_folder)
                        _write_h(target_folder, True, True)
                    except (tarfile.TarError, RuntimeError, KeyboardInterrupt):
                        sys.stderr.write('Decompressing the archive is not success')
                        PrintException()
                        _write_h(target_folder, True, False, filename)
                        if os.path.exists(target_folder):
                            shutil.rmtree(target_folder)
                        raise
                _write_h(target_folder, True, True, filename)
                return True
            else:
                _write_h(target_folder, True, False, filename)
    except Exception as e:
        print(e)
        _write_h(dirname, False, False, filename)
    return False


def pickle_it(file_path, obj):
    """Pickle the obj

    Args:
        file_path (str):
        obj (obj):
    """
    import pickle as pickle
    with open(file_path, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def unpickle(file):
    import _pickle as pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def save_dict_as_h5(save_path, dict_need_save):
    try:
        import h5py
    except ImportError:
        h5py = None
    if h5py is not None:
        with h5py.File(save_path, 'w') as f:
            for k, v in dict_need_save.items():
                f.create_dataset(k, data=v)


def read_dict_from_h5(save_path):
    try:
        import h5py
    except ImportError:
        h5py = None
    if h5py is not None:
        return_dict = OrderedDict()
        with h5py.File(save_path, 'r') as f:
            for k, v in f.items():
                return_dict[k] = v
            return return_dict


def open_pickle(fpath, data_key='data', label_key='labels'):
    d = unpickle(fpath)
    d_decoded = {}
    for k, v in d.items():
        d_decoded[k.decode('utf8')] = v
    d = d_decoded
    data = d[data_key]
    labels = d[label_key]
    return data, labels


def check_image(image, imagepath):
    if not os.path.exists(imagepath): return False
    with open(imagepath, 'rb') as fin:
        return (hashlib.md5(fin.read()).hexdigest() == image['md5'])


def read_mat(mat_path):
    mat = loadmat(mat_path)
    return mat


def get_file_create_time(file_path):
    if platform.system() == 'Windows':
        return os.path.getctime(file_path)
    else:
        stat = os.stat(file_path)
        try:
            return stat.st_birthtime
        except AttributeError:
            # We're probably on Linux. No easy way to get creation dates here,
            # so we'll settle for when its content was last modified.
            return stat.st_mtime
