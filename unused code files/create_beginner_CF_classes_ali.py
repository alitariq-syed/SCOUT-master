# -*- coding: utf-8 -*-
"""
Created on Tue May 18 14:22:54 2021

@author: Ali
"""

### choose random CF classes for each input image


import numpy as np
import pandas as pd
import os

os.chdir('G:\CUB_200_2011\CUB_200_2011')
os.listdir()

np.random.seed(seed = 100)

images = pd.read_csv('images.txt', sep=" ", header=None)#np.loadtxt(fname = 'images.txt')
train_test = pd.read_csv('train_test_split.txt', sep=" ", header=None)
image_class_labels = pd.read_csv('image_class_labels.txt', sep=" ", header=None)

all_correct_random_te=[]# correct predictions
all_predicted_random_te=[]# raw predictions
all_gt_target_random_te = []# actual predictions

num_classes = 200
for i in range(len(train_test)):
    src = './images/'+images[1][i]
    print(i)
    if train_test[1][i] ==1:
        #train image
        continue        
    else:
        #test image
        predict = np.random.randint(num_classes) #class range 0-199
        groundTruth = image_class_labels[1][i] -1#class range 0-199
        
        all_predicted_random_te.append(predict)
        all_gt_target_random_te.append(groundTruth)
        if (predict==groundTruth):
            all_correct_random_te.append(1)
        else:
            all_correct_random_te.append(0)

all_predicted_random_te = np.asarray(all_predicted_random_te)
all_gt_target_random_te = np.asarray(all_gt_target_random_te,dtype='int32')
all_correct_random_te = np.asarray(all_correct_random_te)

##check accuracy 
acc = np.sum(all_predicted_random_te==all_gt_target_random_te)/len(all_predicted_random_te)
print("random accuracy: ", acc*100, " %")

np.save('all_correct_random_te.npy',all_correct_random_te)
np.save('all_predicted_random_te.npy',all_predicted_random_te)
np.save('all_gt_target_random_te.npy',all_gt_target_random_te)

all_correct_student = np.load('all_correct_random_te.npy')
all_predicted_student = np.load('all_predicted_random_te.npy')
all_gt_target_student = np.load('all_gt_target_random_te.npy')