# -*- coding: utf-8 -*-
import os
from sys import argv

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as transforms
import torchvision.models as models

class NIMA(nn.Module):
    """Neural IMage Assessment model by Google"""
    def __init__(self, base_model, num_classes=5):
        super(NIMA, self).__init__()
        #self.features = base_model
        self.features = base_model.features
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.75),
            nn.Linear(in_features=86528, out_features=num_classes),
            nn.Softmax())

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

def score(image,file_name):
    device = torch.device("cpu")
    #base_model = models.alexnet(pretrained=False)
    base_model = models.squeezenet1_1(pretrained=True)
    #modules = list(base_model.children())[:-1] # delete the last fc layer.
    #base_model = nn.Sequential(*modules)
    model = NIMA(base_model)
    model.load_state_dict(torch.load(file_name))
    model.to(device)
    model.eval()
    
    test_transform = transforms.Compose([
    transforms.Resize((224,224)),
    #transforms.RandomCrop(224),
    transforms.ToTensor()])
    image = test_transform(image)
    image = image.unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
    outputs = outputs.view(-1, 5, 1)
    predicted_mean, predicted_std = 0.0, 0.0

    for i in range(5):
        predicted_mean += i * outputs[:,i].cpu()
    for i in range(5):
        predicted_std += outputs[:,i].cpu() * (i - predicted_mean) ** 2
    
    output_score = predicted_mean.numpy().flatten().tolist()[0]
    output_score_std = predicted_std.numpy().flatten().tolist()[0]

    return output_score, output_score_std
    



if __name__ == '__main__':
    main()

