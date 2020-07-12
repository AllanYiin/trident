from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import glob
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import threading
import urllib.request
import warnings
import zipfile
from sys import stderr

import numpy as np
import requests
import six
import tqdm
from scipy.io import loadmat
from six.moves.urllib.error import HTTPError
from six.moves.urllib.error import URLError
from six.moves.urllib.request import urlopen
from tqdm import tqdm
from urllib3.exceptions import NewConnectionError

from trident.backend.common import OrderedDict, PrintException,get_session,make_dir_if_need

try:
    from urllib.request import urlretrieve
except ImportError:
    from six.moves.urllib.request import urlretrieve

def is_connected():
    try:
        urllib.request.urlopen('https://google.com') #Python 3.x
        return True
    except :
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
        self.update(b * bsize - self.n)  # will also set self.n = b * bsize

def calculate_md5(fpath, chunk_size=1024 * 1024):
    md5 = hashlib.md5()
    with open(fpath, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            md5.update(chunk)
    return md5.hexdigest()


def check_md5(fpath, md5, **kwargs):
    return md5 == calculate_md5(fpath, **kwargs)


def check_integrity(fpath, md5=None):
    if not os.path.isfile(fpath):
        return False
    if md5 is None:
        return True
    return check_md5(fpath, md5)

def _write_h(dirname,is_downloaded=False,is_extracted=False):
    _h_path = os.path.join(dirname, 'status.json')
    _h = {
        'is_downloaded': is_downloaded,
        'is_extracted': is_extracted
    }
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

def download_file(src, dirname, filename, desc=''):
    _h = _read_h(dirname)
    if os.path.exists(os.path.join(dirname, filename)) and _h != {} and _h.get('is_downloaded', False) == True:
        print('archive file is already existing, donnot need download again.')
        return True
    else:
        if os.path.exists(os.path.join(dirname, filename)):
            os.remove(os.path.join(dirname, filename))

        try:
            with TqdmProgress(unit='B', unit_scale=True,  leave=True,miniters=10, desc=desc) as t:  # all optional kwargs
                urlretrieve(src, filename=os.path.join(dirname, filename), reporthook=t.update_to, data=None)
                _write_h(dirname, True, False)
            return True
        except Exception as e:
            _write_h(dirname, False, False)
            print('***Cannot download data, so the data provider cannot initialized.\n',flush=True)
            print('***Please check your internet or download  files from following url in another computer, \n and then put them into {0}\n {1} '.format( dirname,src),flush=True)

            print(e)
            return False




def download_file_from_google_drive(file_id, dirname, filename=None, md5=None):
    """Download a Google Drive file from  and place it in root.

    Args:
        file_id (str): id of file to be downloaded
        dirname (str): Directory to place downloaded file in
        filename (str, optional): Name to save the file under. If None, use the id of the file.
        md5 (str, optional): MD5 checksum of the download. If None, do not check
    """
    # Based on https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url
    import requests
    url = "https://docs.google.com/uc?export=download"

    if not filename:
        filename = file_id
    fpath = os.path.join(dirname, filename)
    _h=_read_h(dirname)
    # if os.path.exists(os.path.join(dirname, filename)):
    #     print('archive file is already existing, donnot need download again.')
    if os.path.exists(os.path.join(dirname, filename)) and _h!={} and  _h.get('is_downloaded', False)==True:
        print('archive file is already existing, donnot need download again.')
        return True
    else:
        try:
            session = requests.Session()
            response = session.get(url, params={'id': file_id}, stream=True)
            token = _get_confirm_token(response)
            if token:
                params = {'id': file_id, 'confirm': token}
                response = session.get(url, params=params, stream=True)
            _save_response_content(response, fpath)
            _write_h(dirname,True,False)
            return True
        except Exception as e:
            print('***Cannot download data, so the data provider cannot initialized.\n', flush=True)
            print( '***Please check your internet or download  files from following url in another computer, \n and then put them into {0}\n {1} '.format(dirname, 'https://drive.google.com/open?id={0}'.format(file_id)), flush=True)
            print(e)
            return False

def get_image_from_google_drive(file_id):
    """Download a Google Drive image  and place it in root.

    Args:
        file_id (str): id of file to be downloaded

    Returns:
        the file path of this downloaded image

    """

    import requests
    url = 'https://drive.google.com/uc?export=download'

    filename = file_id

    _session = get_session()
    _trident_dir = _session.trident_dir
    dirname = os.path.join(_trident_dir, 'download')
    make_dir_if_need(dirname)
    fpath = os.path.join(dirname, filename)

    if os.path.exists(fpath) :
        os.remove(fpath)
    try:
        session = requests.Session()
        response = session.get(url, params={'id': file_id}, stream=True)
        content_type=response.headers.get('content-type')
        filename=file_id+'.'+content_type.split('/')[-1]
        fpath = os.path.join(dirname, filename)

        token = _get_confirm_token(response)
        if token:
            params = {'id': file_id, 'confirm': token}
            response = session.get(url, params=params, stream=True)
        _save_response_content(response, fpath)
        return fpath
    except Exception as e:
        print('***Cannot download data, so the data provider cannot initialized.\n', flush=True)
        print( '***Please check your internet or download  files from following url in another computer, \n and then put them into {0}\n {1} '.format(dirname, 'https://drive.google.com/open?id={0}'.format(file_id)), flush=True)
        print(e)
        return None




def download_model_from_google_drive(file_id, dirname, filename=None, md5=None):
    """Download a Google Drive file from  and place it in root.

    Args:
        file_id (str): id of file to be downloaded
        dirname (str): Directory to place downloaded file in
        filename (str, optional): Name to save the file under. If None, use the id of the file.
        md5 (str, optional): MD5 checksum of the download. If None, do not check
    """
    # Based on https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url
    import requests
    url = "https://docs.google.com/uc?export=download"

    if not filename:
        filename = file_id
    fpath = os.path.join(dirname, filename)
    isload = False
    models_md5 = None
    need_download = True

    try:
        session = requests.Session()
        response = session.get(url, params={'id': '12XLjt9Zcaoo90WGG6R5N0U6Sf_KBZZn_'}, stream=False)
        if response.status_code==200:
            if os.path.exists(os.path.join(dirname, 'models_md5.json')):
                os.remove(os.path.join(dirname, 'models_md5.json'))
            with open(os.path.join(dirname, 'models_md5.json'),"wb") as f:
                f.write(response.content)

        check_internet = True
        if os.path.exists(os.path.join(dirname, 'models_md5.json')):
            with open(os.path.join(dirname, 'models_md5.json')) as f:
                models_md5 = json.load(f)
    except Exception:
        PrintException()
        check_internet=False


    if check_internet == False:
        if os.path.exists(os.path.join(dirname, filename)):
            print('internet connect  error,model file is already existing, donnot need download again.')
            return True
        else:
            print('***Cannot download data, so the data provider cannot initialized.\n', flush=True)
            print('***Please check your internet or download  files from following url in another computer, \n and then '
                'put them into {0}\n {1} '.format( dirname, 'https://drive.google.com/open?id={0}'.format(file_id)), flush=True)

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
                session = requests.Session()
                response = session.get(url, params={'id': file_id}, stream=True)
                token = _get_confirm_token(response)
                if token:
                    params = {'id': file_id, 'confirm': token}
                    response = session.get(url, params=params, stream=True)
                _save_response_content(response, fpath)

                if check_integrity(os.path.join(dirname, filename), models_md5[filename]):
                    print('model file is downloaded and validated.')
                else:
                    print('model file is downloaded but not match md5.')
        except Exception as e:
            print('***Cannot download data, so the data provider cannot initialized.\n', flush=True)
            print('***Please check your internet or download  files from following url in another computer, \n and then '
                'put them into {0}\n {1} '.format(dirname, 'https://drive.google.com/open?id={0}'.format(file_id)), flush=True)
            print(e)

        return False


def _get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def _save_response_content(response, destination, chunk_size=32768):
    folder, file=os.path.split(destination)
    progress=0
    with open(destination, "wb") as f:
        pbar = TqdmProgress(response.iter_content(chunk_size=chunk_size), total=None, unit='MB',unit_scale=True, miniters=10,desc=file, leave=True, file=sys.stdout)
        for chunk in pbar:
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
                progress += len(chunk)
                pbar.update(progress - pbar.n)
        pbar.close()




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
  folder,_=os.path.split(file_path)
  _h = _read_h(target_folder)
  if _h != {} and _h.get('is_extracted', False) == True:
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
                _write_h(target_folder, True, False)
                if os.path.exists(target_folder):
                    shutil.rmtree(target_folder)
                raise
        return True
    else:
        _write_h(target_folder, False, False)
  return False


def pickle_it(file_path,dict):
    import pickle as pickle
    with open(file_path, 'wb') as handle:
        pickle.dump(dict, handle, protocol=2)

def unpickle(file):
    import _pickle as pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

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

