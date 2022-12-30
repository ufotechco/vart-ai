# VartVesselPredictor
# Esta clase busca identificar la extensión de vasos sanguíneos
# presente en la imagen entregada
# Desarrollado por UFOTECH S.A.S.
# Basado en Ronneberger, Olaf & Fischer, Philipp & Brox, Thomas. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation

from baseline_aug import get_unet
from glob import glob
from PIL import Image
from skimage.transform import resize
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import os
import cv2
from tensorflow.keras.layers import ReLU
import os
from pathlib import Path
from configparser import ConfigParser

input_config = ConfigParser()
config_path = os.path.join(os.path.dirname(__file__), 'vart_config.ini')
input_config.read(config_path)
base_directory = input_config['DEFAULT']['base_directory']
base_input_directory = input_config['DEFAULT']['base_input_directory']
base_output_directory = input_config['DEFAULT']['base_output_directory']
model_path = input_config['VESSEL_DETECT']['model_path']


batchsize = 4
input_shape = (576, 576)

os.environ['CUDA_VISIBLE_DEVICES'] = "0"


def batch(iterable, n=batchsize):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def read_input(path):
    x = np.array(Image.open(path))/255.
    return x


def read_gt(path):
    x = np.array(Image.open(path))
    return x[..., np.newaxis]/np.max(x)

def im_resizing(img_,img):
    img_or = cv2.imread(img_, cv2.IMREAD_UNCHANGED)
    width = img_or.shape[1]
    height = img_or.shape[0]
    dim = (width, height)
    
    # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return resized
    
if __name__ == '__main__':
    model_name = "baseline_unet_aug_do_0.1_activation_ReLU_"

    val_data = list(zip(sorted(glob(base_output_directory+'improved/*.png')),
                          sorted(glob(base_input_directory+'2nd_manual/*.gif')),
                        sorted(glob(base_input_directory+'mask/*.gif'))))

    model = get_unet(do=0.1, activation=ReLU)

    file_path = base_directory+model_name + "weights.best.hdf5"
    
    model.load_weights(model_path, by_name=True)

    gt_list = []
    pred_list = []

    for batch_files in tqdm(batch(val_data), total=len(val_data)//batchsize):

        imgs = [resize(read_input(image_path[0]), input_shape) for image_path in batch_files]
        seg = [read_gt(image_path[1]) for image_path in batch_files]
        mask = [read_gt(image_path[2]) for image_path in batch_files]

        imgs = np.array(imgs)

        pred = model.predict(imgs)

        pred_all = (pred)

        pred = np.clip(pred, 0, 1)

        for i, image_path in enumerate(batch_files):

            pred_ = pred[i, :, :, 0]

            pred_ = resize(pred_, (584, 565))

            mask_ = mask[i]

            gt_ = (seg[i]>0.5).astype(int)

            gt_flat = []
            pred_flat = []

            for p in range(pred_.shape[0]):
                for q in range(pred_.shape[1]):
                    if mask_[p,q]>0.5: # Inside the mask pixels only
                        gt_flat.append(gt_[p,q])
                        pred_flat.append(pred_[p,q])

            gt_list += gt_flat
            pred_list += pred_flat

            pred_ = 255.*(pred_ - np.min(pred_))/(np.max(pred_)-np.min(pred_))

            image_base = Path(image_path[0]).stem
            image_ext = Path(image_path[0]).suffix
            
            cv2.imwrite(base_output_directory+"noir/"+image_base+"-noir"+image_ext, im_resizing(image_path[0],pred_))
