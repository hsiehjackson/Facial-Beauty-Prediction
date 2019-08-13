# [107-1] Cognitive Computing Final - Facial Beauty Prediction
In this project, we are trying to develop a **facial beauty prediction** framework based on [Paper](https://github.com/abishekarun/Facial-Beauty-Prediction/blob/master/SCUT_FBP5500.pdf). We estimate the attractiveness rating of faces from [SCUT_FBP5500](https://drive.google.com/file/d/1w0TorBfTIqbquQVd6k3h_77ypnrvfGwf/view) dataset and build a **live demo system** to view the results with a laptop camera.

## How to Use
* Git clone the code
```
git clone https://github.com/hsiehjackson/Facial-Beauty-Prediction.git
```
* Download the dataset and extract zip file 
```
bash download.sh
unzip dataset.zip
```
* Annotation for cross validation
```
python src/annotation.py
```
* Training on EMD/MSE/BCE Loss
```
python src/main_[emd/mse/bce].py --use_model=[YOU CAN SELECT]
```
* Testing on one validation [1-5]
```
python src/main_[emd/mse/bce].py --MODE=test --load_cv=[1-5] --use_model=[YOU CAN SELECT] --load_model=[Model ckpt path]
```

* Testing on Live Demo (SqueezeNet)
```
cd test/
[Single Image] python test_image.py [Image File]
[Grid Image] python test_grid.py [Image File] --mode=[mean/std]
[Video] python test_video.py (assume you have a camera!!!)
```
## Dataset Introduction
The SCUT-FBP5500 Dataset can be divided into four subsets with different races and gender, including Asian females/males and Caucasian females/males. All the images are labeled with beauty scores ranging from [1, 5] by totally 60 volunteers. 

<img src="https://i.imgur.com/PZDOqVG.png" alt="drawing"/> 

We analyzed the beauty scores distribution [0, 4] and find that the **extreme high/low scores or mean scores have small variance**. The results meet the common sense, which we may share similar feelings about beauty on special or normal images. From the below picture, we can also find voluteers tend to give scores **lower than average 2**, which we show in green color.

<img src="https://i.imgur.com/z1htRxD.png"
alt="drawing"/> 

## Proposed Methods
To learn the distribution of beauty feelings from survey, we cannot view the scores as independent class. Therefore, we try to use the Earth Mover Distance-based loss based on [Paper](https://arxiv.org/pdf/1611.05916.pdf) to deal with the **class relationship problems**. The results were compared with EMD Loss, Mean Square Error, and Multi-Binary Cross Entropy Loss as following.

* EMD Loss: 
    * Learn distribution of beauty score
    * Consider class relationship
* Mean Square Error:
    * Just learn the mean value of beauty score
    * Cannot analyze the distribution of beauty score
* Multi-Binary Cross Entropy Loss:
    * Split different scores as an independent multiclass problems
    * Use cross entropy to learn class distribution independently

## Metrics Results
We tested the loss with different ImageNet models and showed the results as following. The experiments were all applied on **five fold cross validation** schemes. Several conclusions we can discuss:
* Complexed model (more parameters) has higher performance
* EMD Loss has the highest performance considering overall metrics

> pearson correlation (PC)
> maximum absolute error (MAE) 
> root mean square error (RMSE)

<img src="https://i.imgur.com/PxHeLGF.png"
alt="drawing"/> 

## Images Results
From the following images results, we can assume our model is **sensitive to light**, such as white and black color. However, we would obtained lower scores for the brightest images regarding different value and statuation (HSV).

Mean(1.88) Std(1.03)           |  Mean(2.0) Std(0.91) 
:-------------------------:|:-------------------------:
<img src="https://i.imgur.com/1e0tNd1.jpg" alt="drawing" width="600" height="250"/>  | <img src="https://i.imgur.com/ZH9IQ5G.jpg" alt="drawing" width="600" height="250"/>

Mean(2.46) Std(0.52)           |  Mean(2.85) Std(0.48) 
:-------------------------:|:-------------------------:
<img src="https://i.imgur.com/Mn8AUai.jpg" alt="drawing" width="600" height="250"/>  | <img src="https://i.imgur.com/0H2lUeV.jpg" alt="drawing" width="600" height="250"/>

Mean(1.65) Std(0.63)           |  Mean(2.23) Std(0.68) 
:-------------------------:|:-------------------------:
<img src="https://i.imgur.com/9f2y9ER.jpg" alt="drawing" width="600" height="150"/>  | <img src="https://i.imgur.com/N0JS2Mf.jpg" alt="drawing" width="600" height="200"/>

Mean           |  Std
:-------------------------:|:-------------------------:
<img src="https://i.imgur.com/vfZdZQR.jpg" alt="drawing" width="600" height="300"/> | <img src="https://i.imgur.com/2Msh4rr.jpg" alt="drawing" width="600" height="300"/> 



