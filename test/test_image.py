import cv2
import os
from sys import argv
from PIL import Image, ImageDraw, ImageFont
from score import *
import matplotlib.pyplot as plt
import warnings
import argparse


parser = argparse.ArgumentParser(description='parser')
parser.add_argument('image_file', type=str)
parser.add_argument('--model',type=str, default='model/squeeze-0.218914.pkl')
parser.add_argument('--font',type=str,default='font/Verdana.ttf')
parser.add_argument('--font_size',type=int,default=50)
args = parser.parse_args()


def main():
    image = Image.open(args.image_file)
    mean, std = score(image,args.model)
    mean = round(mean, 2)
    std = round(std,2)
    print('Load: {} | score: {} % {}'.format(args.image_file,mean, std))
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(args.font, args.font_size)
    word = "{}%{}".format(mean, std)
    text_w, text_h = draw.textsize(word, font)
    img_w, img_h = image.size
    draw.text((int((img_w-text_w)/2),img_h-text_h),word, font=font,fill=(0, 0, 255))
    plt.imshow(image)
    plt.xticks([])
    plt.yticks([])
    plt.show()
if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    main()