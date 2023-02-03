import os
from yacs.config import CfgNode

_C = CfgNode()

_C.DATASET_DIR = 'foocus_dataset'
_C.IMAGE_DIR = os.path.join(_C.DATASET_DIR, 'flickr_logos_27_dataset_images')
_C.ANNOT_FILE = os.path.join(
    _C.DATASET_DIR, 'foocus_dataset_training_set_annotation.tx')
_C.CROPPED_ANNOT_FILE = os.path.join(
    _C.DATASET_DIR, 'foocus_dataset_training_set_annotation_cropped.txt')
_C.CROPPED_ANNOT_FILE_TEST = os.path.join(
    _C.DATASET_DIR, 'foocus_dataset_test_set_annotation_cropped.txt')

# _C.CLASS_NAMES = [
#     'Adidas', 'Apple', 'BMW', 'Citroen', 'Cocacola', 'DHL', 'Fedex', 'Ferrari',
#     'Ford', 'Google', 'HP', 'Heineken', 'Intel', 'McDonalds', 'Mini', 'Nbc',
#     'Nike', 'Pepsi', 'Porsche', 'Puma', 'RedBull', 'Sprite', 'Starbucks',
#     'Texaco', 'Unicef', 'Vodafone', 'Yahoo'
# ]


_C.CLASS_NAMES = [
    'Norsk Tipping', 'Rema1000', 'Tess'
]

def get_cfg_defaults():
    return _C.clone()
