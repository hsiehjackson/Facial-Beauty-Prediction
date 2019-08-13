import cv2
import os
from PIL import Image, ImageDraw, ImageFont
from score import *
import matplotlib.pyplot as plt
import warnings
import torchvision as vis
import torch
import argparse



parser = argparse.ArgumentParser(description='parser')
parser.add_argument('image_file', type=str)
parser.add_argument('--mode', type=str, default='mean', help=['mean','std'])
parser.add_argument('--model',type=str, default='model/squeeze-0.218914.pkl')
parser.add_argument('--font',type=str,default='font/Verdana.ttf')
parser.add_argument('--font_size',type=int,default=200)
args = parser.parse_args()


def modified(image, prop, dim):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    #h, s, v = cv2.split(hsv) #h:180, s:256, v:256
    v = hsv[:,:,1]
    s = hsv[:,:,2]
    if dim == 's':
        s = s * (1 + prop)
        s[s > 255] = 255
        hsv[:,:,2] = s
    elif dim == 'v':
        v = v * (1 + prop)
        v[v > 255] = 255
        hsv[:,:,1] = v
    #final_hsv = cv2.merge((h,s,v))
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return image

def getscore(image):
    white = Image.fromarray(np.full_like(image,0))
    image = Image.fromarray(image)
    mean, std = score(image, args.model)
    mean = round(mean, 2)
    std = round(std,2)
    #draw = ImageDraw.Draw(white)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(args.font, args.font_size)
    if args.mode == 'mean':
        word = "{}".format(mean)
    elif args.mode == 'std':
        word = "{}".format(std)
    text_w, text_h = draw.textsize(word, font)
    img_w, img_h = image.size
    draw.text((int((img_w-text_w)/2),int(img_h-text_h)/2),word, font=font,fill=(0, 0, 255))
    #white = np.array(white)
    image = np.array(image)
    return image

def main():
    img = cv2.imread(args.image_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_all = []
    for s in np.arange(-0.3,0.4,0.1):
        img_s = modified(img,s,dim='s')
        for v in np.arange(-0.3,0.4,0.1):
            final_img = modified(img_s,v,dim='v')
            final_img = getscore(final_img)
            img_all.append(final_img)
    img_all = np.array(img_all)
    img_all = np.transpose(img_all,(0,3,1,2))
    print(img_all.shape)
    img_all = torch.from_numpy(img_all)
    img_all = vis.utils.make_grid(tensor=img_all,nrow=7)
    img_all = img_all.numpy()
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('saturation')
    plt.ylabel('value')
    plt.title('image')
    plt.imshow(np.transpose(img_all, (1,2,0)),interpolation='nearest')
    if args.mode == 'mean':
        plt.savefig('mean_{}.jpg'.format(args.image_file))
    elif args.mode == 'std':
        plt.savefig('std_{}.jpg'.format(args.image_file))
    plt.show()

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    main()