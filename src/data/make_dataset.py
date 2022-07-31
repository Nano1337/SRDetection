# -*- coding: utf-8 -*-
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import os

# image processing libraries
import cv2
import numpy as np

# import deep learning libraries
import torch

# import utils
from utils import normalize_01, re_normalize, UNet

# preprocess function
def preprocess(img: np.ndarray):
    img = np.moveaxis(img, -1, 0)  # from [H, W, C] to [C, H, W]
    img = normalize_01(img)  # linear scaling to range [0-1]
    img = np.expand_dims(img, axis=0)  # add batch dimension [B, C, H, W]
    img = img.astype(np.float32)  # typecasting to float32
    return img

# postprocess function
def postprocess(img: torch.tensor):
    img = img.cpu().numpy()  # send to cpu and transform to numpy.ndarray
    img = np.squeeze(img)  # remove batch dim and channel dim -> [H, W]
    img = re_normalize(img)  # scale it to the range [0-255]
    return img

def predict(img,
            model,
            preprocess,
            postprocess,
            device,
            ):
    model.eval()
    img = preprocess(img)  # preprocess image
    x = torch.from_numpy(img).to(device)  # to torch, send to device
    with torch.no_grad():
        out = model(x)  # send through model/network
    out_sigmoid = torch.sigmoid(out)  # perform softmax on outputs
    result = postprocess(out_sigmoid)  # postprocess outputs
    return result

def reflection_enhance(img):
    ''' Enhance image to improve specular reflection contrast from other parts of image'''
    norm_img = np.float32(cv2.normalize(img.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX))
    hsv = cv2.cvtColor(norm_img, cv2.COLOR_RGB2HSV) 
    s = hsv[:, :, 1]
    for i in range(3):
        norm_img[:, :, i] = (1 - s) * norm_img[:, :, i]
    return norm_img

def make_output(img, model, device):
    """
    Generate the output image for the given image
    """

    # calculate enhanced image
    enhanced = reflection_enhance(img)*255

    # DL model prediction 
    output1 = predict(enhanced, model, preprocess, postprocess, device)

    output1[output1!=0] = 1

    # get grayscale from enhanced image V channel of HSV
    gray = enhanced[:, :, 2]

    # making mask
    output2 = np.zeros((img.shape[0], img.shape[1]))
    output2[gray>194] = 1

    # combine masks
    output = output1 + output2
    output[output == 2] = 1

    return output

def load_model():
    # device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # model
    model = UNet(in_channels=3,
                out_channels=1,
                n_blocks=4,
                start_filters=8,
                activation='relu',
                normalization='batch',
                conv_mode='same',
                dim=2).to(device)


    model_name = 'unetv5.pt'
    model_weights = torch.load(r"C:\Users\haoli\OneDrive\Documents\SRDetection\models\{}".format(model_name))
    model.load_state_dict(model_weights)

    return model, device

def analyze_results(output_img, output_mask, save_path):
    """ Checks the results of the model and saves them to a folder"""
    logger = logging.getLogger(__name__)
    logger.info('analyzing results')
    for i in range(len(os.listdir(output_img))):
        img_path = os.path.join(output_img, str(i).zfill(5) + '.png')
        img = cv2.imread(img_path)
        mask_path = os.path.join(output_mask, str(i).zfill(5) + '.png')
        output = cv2.imread(mask_path)
        output = output[:, :, 0]
        img[output == 255] = 0
        img_output_path = os.path.join(save_path, str(i).zfill(5) + '.png')
        cv2.imwrite(img_output_path, img)
        logger.info('{} processed'.format(img_path))

    print("Done")

def main(input_filepath, img_output, mask_output):
    """ Runs data processing scripts to turn raw data into
        cleaned data ready to be analyzed.
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    # load model
    model, device = load_model()

    count = 0
    for folder in os.listdir(input_filepath):
        for img_file in os.listdir(os.path.join(input_filepath, folder)):
            img_path = os.path.join(input_filepath, folder, img_file)
            img = cv2.imread(img_path)
            output = make_output(img, model, device)*255
            img_output_path = os.path.join(img_output, str(count).zfill(5) + '.png')
            mask_path = os.path.join(mask_output, str(count).zfill(5) + '.png')
            cv2.imwrite(img_output_path, img)
            cv2.imwrite(mask_path, output)
            logger.info('{} processed'.format(img_path))
            count += 1

    print('{} images processed'.format(count))


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    # define input and output paths
    input_dir = Path(r"D:\GLENDA_v1.5_no_pathology\no_pathology\frames")
    output_img = Path(r"D:\GLENDA_v1.5_no_pathology\no_pathology\GLENDA_img")
    output_mask = Path(r"D:\GLENDA_v1.5_no_pathology\no_pathology\GLENDA_mask")

    # main(input_dir, output_img, output_mask)
    analyze_output_path = Path(r"D:\GLENDA_v1.5_no_pathology\no_pathology\GLENDA_analyze")
    analyze_results(output_img, output_mask, analyze_output_path)
