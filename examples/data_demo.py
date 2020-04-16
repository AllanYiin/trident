import os
os.environ['TRIDENT_BACKEND'] = 'cntk'

from  trident import data

raw_imgs,raw_labels=data.load_stanford_cars('cars','train')

