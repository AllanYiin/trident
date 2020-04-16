'''
 utils for loading dataset
'''


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import threading
import os
import subprocess
import sys
import glob
import tqdm
from  sys import stderr
import requests
import warnings
import numpy as np
from tqdm import tqdm
import gzip
import tarfile
import zipfile
import shutil
import hashlib
import glob
import six
from scipy.io import loadmat

from six.moves.urllib.error import HTTPError
from six.moves.urllib.error import URLError
from six.moves.urllib.request import urlopen
try:
    from urllib.request import urlretrieve
except ImportError:
    from six.moves.urllib.request import urlretrieve



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


def download_file(src, dirname, filename, desc=''):
    if os.path.exists(os.path.join(dirname, filename)):
        print('archive file is already existing, donnot need download again.')
    else:
        error_msg = 'URL fetch failure on {}: {} -- {}'
        try:
            try:
                with TqdmProgress(unit='B', unit_scale=True, miniters=10, desc=desc) as t:  # all optional kwargs
                    urlretrieve(src, filename=os.path.join(dirname, filename), reporthook=t.update_to, data=None)
            except HTTPError as e:
                raise Exception(error_msg.format(src, os.path.join(dirname, filename), e.code, e.__str__()))
            except URLError as e:
                raise Exception(error_msg.format(src, os.path.join(dirname, filename), e.errno, e.reason))
            except (Exception, KeyboardInterrupt) as e:
                if os.path.exists(os.path.join(dirname, filename)):
                    os.remove(os.path.join(dirname, filename))
        except:
            raise


def extract_archive(file_path, path='.', archive_format='auto'):
  """Extracts an archive if it matches tar, tar.gz, tar.bz, or zip formats.

  Arguments:
      file_path: path to the archive file
      path: path to extract the archive file
      archive_format: Archive format to try for extracting the file.
          Options are 'auto', 'tar', 'zip', and None.
          'tar' includes tar, tar.gz, and tar.bz files.
          The default 'auto' is ['tar', 'zip'].
          None or an empty list will return no matches found.

  Returns:
      True if a match was found and an archive extraction was completed,
      False otherwise.
  """
  if archive_format is None:
    return False
  if archive_format == 'auto':
    archive_format = ['tar', 'zip']
  if isinstance(archive_format, six.string_types):
    archive_format = [archive_format]

  for archive_type in archive_format:
    if archive_type == 'tar':
      open_fn = tarfile.open
      is_match_fn = tarfile.is_tarfile
    if archive_type == 'zip':
      open_fn = zipfile.ZipFile
      is_match_fn = zipfile.is_zipfile

    if is_match_fn(file_path):
      with open_fn(file_path) as archive:
        try:
          if os.path.exists(path):
              print('extracted folder is already existing, donnot need extract again.')
          else:
            archive.extractall(path)
        except (tarfile.TarError, RuntimeError, KeyboardInterrupt):
          if os.path.exists(path):
            if os.path.isfile(path):
              os.remove(path)
            else:
              shutil.rmtree(path)
          raise
      return True
  return False


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

