# -*- coding: utf-8 -*-
import os
from sys import argv

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import PIL

import torch
import torch.autograd as autograd
import torch.optim as optim

import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torchvision.models as models

from logger import Logger

from data_loader import Dataset
from model import *
from util import *
from path import *
import argparse
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='parser')

parser.add_argument('--MODE',type=str,default='train',help=['train','test'])
parser.add_argument('--plot',type=bool,default=False,help='View each image in test mode')
parser.add_argument('--load_model',type=str)
parser.add_argument('--load_cv',type=str)

parser.add_argument('--use_model',type=str,default='squeezenet',help=['squeezenet','alexnet','resnet18','resnet50'])

parser.add_argument('--EPOCHS',type=int,default=200)
parser.add_argument('--CONV_LR',type=int,default=1e-4)
parser.add_argument('--DENSE_LR',type=int,default=1e-4)
parser.add_argument('--train_batch_size',type=int,default=32)
parser.add_argument('--val_batch_size',type=int,default=32)
parser.add_argument('--test_batch_size',type=int,default=1)
parser.add_argument('--num_workers',type=int,default=0)
parser.add_argument('--save_fig',type=bool,default=True,help='Save training progress')
parser.add_argument('--device',type=int,default=0)
args = parser.parse_args()

torch.cuda.set_device(args.device)

def main():
    if args.MODE == 'train':
        for CV in range(1,6):
            train_csv_file = train_csv_path.format(CV)
            val_csv_file = val_csv_path.format(CV)
            file_name = 'cv{}_mse_{}'.format(CV,args.use_model)
            train(CV, train_csv_file, val_csv_file, file_name)
    elif args.MODE == 'test':
        CV = args.load_cv
        val_csv_file = val_csv_path.format(CV)
        test(CV,val_csv_file)


def select_model(device):
    if args.use_model == 'squeezenet':
        base_model = models.squeezenet1_1(pretrained=True)
    elif args.use_model == 'resnet18':
        base_model = models.resnet18(pretrained=True)
    elif args.use_model == 'resnet50':
        base_model = models.resnet50(pretrained=True)
    elif args.use_model == 'alexnet':
        base_model = models.alexnet(pretrained=True)
        modules = list(base_model.children())[:-1] # delete the last fc layer.
        base_model = nn.Sequential(*modules)
    model = Regression(base_model).to(device)
    return model

def train(CV,train_csv_file,val_csv_file,file_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = select_model(device)
    loss_fn = torch.nn.MSELoss()

    train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10, resample=PIL.Image.BILINEAR),
    transforms.ToTensor()])
    
    val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.ToTensor()])

    os.makedirs(os.path.join(ckpt_path,file_name),exist_ok=True)
    save_path = os.path.join(ckpt_path,file_name)
    logger = Logger(save_path)

    trainset = Dataset(csv_file=train_csv_file, root_dir=img_path, transform=train_transform)
    valset = Dataset(csv_file=val_csv_file, root_dir=img_path, transform=val_transform)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.train_batch_size,
        shuffle=True, num_workers=args.num_workers)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=args.val_batch_size,
        shuffle=False, num_workers=args.num_workers)

    optimizer = optim.Adam([{'params': model.features.parameters(), 'lr': args.CONV_LR},
                            {'params': model.classifier.parameters(), 'lr': args.DENSE_LR}])

    # send hyperparams
    info = ({
        'train_batch_size': args.train_batch_size,
        'val_batch_size': args.val_batch_size,
        'conv_base_lr': args.CONV_LR,
        'dense_lr': args.DENSE_LR,
        })

    for tag, value in info.items():
        logger.scalar_summary(tag, value, 0)

    param_num = 0
    for param in model.parameters():
        param_num += int(np.prod(param.shape))
    print('Trainable params: %.2f million' % (param_num / 1e6))
    #loss_fn = softCrossEntropy()

    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    train_pc, val_pc = [], []
    train_mae, val_mae = [], []
    train_rmse, val_rmse = [], []
    for epoch in range(0, args.EPOCHS):
        start = time.time()
        for batch_idx, data in enumerate(train_loader):
            images = data['image'].to(device)
            labels = data['annotations'].to(device).float()
            model.train()
            outputs = model(images)
            optimizer.zero_grad()

            target_mean = 0.0
            for i in range(5):
                target_mean += i * labels[:,i].cpu()
            if batch_idx == 0:
                predicted = outputs.cpu()
                target = target_mean
            else:
                predicted = torch.cat((predicted, outputs.cpu()), 0)
                target = torch.cat((target, target_mean), 0)

            PC = pearsonr_loss(target_mean, outputs.cpu())
            MAE = MAE_loss(target_mean, outputs.cpu())
            RMSE = RMSE_loss(target_mean, outputs.cpu())
            loss = loss_fn(target_mean.to(device).float(), outputs)

            loss.backward()
            optimizer.step()

            if batch_idx > 0:                
                print('\rCV{} Epoch: {}/{} | MSE: {:.4f} | PC: {:.4f} | MAE: {:.4f} | RMSE: {:.4f} | [{}/{} ({:.0f}%)] | Time: {}  '.format(CV ,epoch + 1, args.EPOCHS, 
                loss, PC, MAE, RMSE, batch_idx*args.train_batch_size, len(train_loader.dataset),
                100.*batch_idx*args.train_batch_size/len(train_loader.dataset), 
                timeSince(start, batch_idx*args.train_batch_size/len(train_loader.dataset))), end='')
            
        train_losses.append(loss_fn(target, predicted))
        train_pc.append(pearsonr_loss(target, predicted))
        train_mae.append(MAE_loss(target, predicted))
        train_rmse.append(RMSE_loss(target, predicted))

        # do validation after each epoch
        for batch_idx, data in enumerate(val_loader):
            images = data['image'].to(device)
            labels = data['annotations'].to(device).float()
            with torch.no_grad():
                model.eval()
                outputs = model(images)
            target_mean = 0.0
            for i in range(5):
                target_mean += i * labels[:,i].cpu()
            if batch_idx == 0:
                predicted = outputs.cpu()
                target = target_mean
            else:
                predicted = torch.cat((predicted, outputs.cpu()), 0)
                target = torch.cat((target, target_mean), 0)

        val_losses.append(loss_fn(target, predicted))
        val_pc.append(pearsonr_loss(target, predicted))
        val_mae.append(MAE_loss(target, predicted))
        val_rmse.append(RMSE_loss(target, predicted))
        
        info = {'conv_base_lr': args.CONV_LR,
                'dense_lr': args.DENSE_LR,
                'train CE loss': train_losses[-1],
                'train mae loss': train_mae[-1],
                'train rmse loss' : train_rmse[-1],
                'train pc': train_pc[-1],
                'val CE loss': val_losses[-1],
                'val mae loss': val_mae[-1],
                'val rmse loss' : val_rmse[-1],
                'val pc': val_pc[-1]}
        for tag, value in info.items():
            logger.scalar_summary(tag, value, epoch+1)

        print('\ntrain MSE %.4f | train PC %.4f | train MAE %.4f | train RMSE: %.4f' % (train_losses[-1], train_pc[-1], train_mae[-1], train_rmse[-1]))
        print('valid MSE %.4f | valid PC %.4f | valid MAE %.4f | valid RMSE: %.4f' % (val_losses[-1], val_pc[-1], val_mae[-1], val_rmse[-1]))

        # Use early stopping to monitor training
        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            # save model weights if val loss decreases
            torch.save(model.state_dict(), os.path.join(save_path, 'MSE-%f.pkl' % (best_val_loss)))
            print ('Save Improved Model(MSE_loss = %.6f)...' % (best_val_loss))
            # reset stop_count
        if args.save_fig and (epoch+1) % 100 == 0:
            epochs = range(1, epoch + 2)
            plt.plot(epochs, train_losses, 'b-', label='train MSE')
            plt.plot(epochs, val_losses, 'g-', label='val MSE')
            plt.plot(epochs, train_pc, 'r-', label='train pc')
            plt.plot(epochs, val_pc, 'y', label='val pc')
            plt.title('MSE loss')
            plt.legend()
            plt.savefig(save_path+'loss.png')

def test(CV,val_csv_file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = select_model(device)
    model.load_state_dict(torch.load(args.load_model))
    model.to(device)
    model.eval()
    val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.ToTensor()])
    testset = Dataset(csv_file=val_csv_file, root_dir=img_path, transform=val_transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers)
    print('***CV_{}***'.format(CV))
    for batch_idx, data in enumerate(test_loader):
        print('image count: {}'.format((batch_idx+1)*args.test_batch_size),end='\r')
        image = data['image'].to(device)
        labels = data['annotations'].to(device).float()
        with torch.no_grad():
            output = model(image)
        target_mean, target_std = 0.0, 0.0
        for i in range(5):
            target_mean += i * labels[:,i].cpu()
        #target_mean = target_mean.view(target_mean.numel())
        for i in range(5):
            target_std += labels[:,i].cpu() * (i - target_mean) ** 2

        if batch_idx == 0:
            predicted = output.cpu()
            target = target_mean
        else:
            predicted = torch.cat((predicted, output.cpu()), 0)
            target = torch.cat((target, target_mean), 0)
        if args.PLOT:
            output_score = output.cpu().numpy().flatten().tolist()[0]
            target_score = target_mean.numpy().flatten().tolist()[0]
            target_score_std = target_std.numpy().flatten().tolist()[0]
            img = data['image'].cpu().squeeze(0).permute(1,2,0).numpy()
            print('beauty score: {:.2f}({:.2f}%{:.2f})'.format(output_score, target_score, target_score_std))
            plt.imshow(img)
            plt.show()
 
    print('MSE LOSS: {:.4f}'.format(loss_fn(target, predicted)))
    print('PC: {:.4f}'.format(pearsonr_loss(target, predicted)))
    print('MAE LOSS: {:.4f}'.format(MAE_loss(target, predicted)))
    print('RMSE LOSS: {:.4f}'.format(RMSE_loss(target, predicted)))
    print('\n')
if __name__ == '__main__':
    main()

