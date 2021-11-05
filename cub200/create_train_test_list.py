# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 10:27:44 2020

@author: Ali
"""

import numpy as np
import pandas as pd
import os

dataset_path = 'D:\Ali Tariq\SCOUT-master\cub200\CUB_200_2011'

os.chdir(dataset_path)
os.listdir()



images = pd.read_csv('images.txt', sep=" ", header=None)#np.loadtxt(fname = 'images.txt')
train_test = pd.read_csv('train_test_split.txt', sep=" ", header=None)

classes = pd.read_csv('image_class_labels.txt', sep=" ", header=None)


train_list=[]
test_list=[]

train_ind=0
test_ind = 0

#impath, imlabel, imindex
for i in range(len(train_test)):
    src = './images/'+images[1][i]
    print(i)
    if train_test[1][i] ==1:
        #train image
        train_list.append((dataset_path+'/images/'+images[1][i], classes[1][i],train_ind))
        train_ind = train_ind+1
    else:
        #test image
        test_list.append((dataset_path+'/images/'+images[1][i], classes[1][i],test_ind))
        test_ind = test_ind+1


train_list = np.asarray(train_list)
test_list = np.asarray(test_list)

np.savetxt('CUB200_gt_tr.txt', train_list,fmt ='%s,%s,%s',delimiter=",")
np.savetxt('CUB200_gt_te.txt', test_list,fmt ='%s,%s,%s',delimiter=",")

# with open('train_list.txt', 'w') as fp:
#     for i in train_list:
#         fp.write(str(i[0]),',',str(i[1]),',',str(i[2]))
#         fp.write('\n')
# with open('test_list.txt', 'w') as fp:
#     for i in test_list:
#         fp.write(str(i))
#         fp.write('\n')
# fp.close()

