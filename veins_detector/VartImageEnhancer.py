# VartImageEnhancer
# Esta clase ejecuta una mejora en la calidad de la imagen
# basado en la red convolucional eficiente del subpixel
# Desarrollado y adaptado por UFOTECH SAS
# Fuente: Shi et al. 2016

import tensorflow as tf
import os
import numpy as np
import PIL
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
import argparse
from configparser import ConfigParser
import sys

input_config = ConfigParser()
config_path = os.path.join(os.path.dirname(__file__), 'config.ini')
input_config.read(config_path)
base_directory = input_config['DEFAULT']['base_directory']
base_output_directory = input_config['DEFAULT']['base_output_directory']
improve_iter = int(input_config['IMAGE_IMPROVE']['improve_iter'])
model_path = input_config['IMAGE_IMPROVE']['model_path']
default_file = input_config['IMAGE_IMPROVE']['default_file']

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=str, default=default_file, help='Image file to be improved')
args = vars(parser.parse_args())
input_file = args["source"]

loss_fn = keras.losses.MeanSquaredError()
optimizer = keras.optimizers.Adam(learning_rate=0.001)


def upscale_image(model, img):
    """Predict the result based on input image and restore the image as RGB."""
    ycbcr = img.convert("YCbCr")
    y, cb, cr = ycbcr.split()
    y = img_to_array(y)
    y = y.astype("float32") / 255.0

    input = np.expand_dims(y, axis=0)
    out = model.predict(input)

    out_img_y = out[0]
    out_img_y *= 255.0

    # Restore the image in RGB color space.
    out_img_y = out_img_y.clip(0, 255)
    out_img_y = out_img_y.reshape((np.shape(out_img_y)[0], np.shape(out_img_y)[1]))
    out_img_y = PIL.Image.fromarray(np.uint8(out_img_y), mode="L")
    out_img_cb = cb.resize(out_img_y.size, PIL.Image.Resampling.BICUBIC)
    out_img_cr = cr.resize(out_img_y.size, PIL.Image.Resampling.BICUBIC)
    out_img = PIL.Image.merge("YCbCr", (out_img_y, out_img_cb, out_img_cr)).convert(
        "RGB"
    )
    return out_img


def get_lowres_image(img, upscale_factor):
    """Return low-resolution image to use as model input."""
    return img.resize(
        (img.size[0] // upscale_factor, img.size[1] // upscale_factor),
        PIL.Image.Resampling.BICUBIC,
    )

total_bicubic_psnr = 0.0
total_test_psnr = 0.0
upscale_factor=3

new_model= keras.models.load_model(model_path, compile=False)
new_model.compile(
    optimizer=optimizer, loss=loss_fn,
)


img = load_img(input_file)
im1_dir =base_output_directory+'improved/'+os.path.split(input_file)[1]
im1 = img.save(im1_dir)

for i in range(0,improve_iter):
    img = load_img(im1_dir)
    lowres_input = get_lowres_image(img, upscale_factor)
    w = lowres_input.size[0] * upscale_factor
    h = lowres_input.size[1] * upscale_factor
    highres_img = img.resize((w, h))
    prediction = upscale_image(new_model, lowres_input)
    lowres_img = lowres_input.resize((w, h))
    lowres_img_arr = img_to_array(lowres_img)
    highres_img_arr = img_to_array(highres_img)
    predict_img_arr = img_to_array(prediction)
    bicubic_psnr = tf.image.psnr(lowres_img_arr, highres_img_arr, max_val=255)
    test_psnr = tf.image.psnr(predict_img_arr, highres_img_arr, max_val=255)

    total_bicubic_psnr += bicubic_psnr
    total_test_psnr += test_psnr

    im1 = highres_img.save(im1_dir)



