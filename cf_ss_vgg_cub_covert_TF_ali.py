import argparse
import os
import shutil
import time

# import torch
# import torch.nn as nn
# import torch.nn.parallel
# import torch.backends.cudnn as cudnn
# import torch.optim
# import torch.utils.data
import numpy as np
#import datasets
#import models as models
import matplotlib.pyplot as plt
# import torchvision.models as torch_models
#from extra_setting import *
# from torch.autograd import Variable
# from torch.autograd import Function
# from torchvision import utils
import scipy.io as sio
from sklearn.svm import SVR
# from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import learning_curve
# from sklearn.kernel_ridge import KernelRidge
import cv2
import seaborn as sns
import operator
import timeit

import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16,decode_predictions, preprocess_input
from tensorflow.keras.datasets import mnist, cifar10
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten,Softmax, GlobalAveragePooling2D
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.losses import categorical_crossentropy
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import math
#from tf_explain_modified.core.grad_cam import GradCAM
import datetime
from tqdm import tqdm # to monitor progress
import argparse
import os, sys

from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import optimizers


# model_names = sorted(name for name in models.__dict__
#                      if name.islower() and not name.startswith("__")
#                      and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch end2end cub200 Training')
parser.add_argument('-d', '--dataset', default='cub200', help='dataset name')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20',
                    choices='model_names',
                    help='model architecture: ' +
                         ' | '.join('model_names') +
                         ' (default: resnet20)')
parser.add_argument('-c', '--channel', type=int, default=16,
                    help='first conv channel (default: 16)')
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers (default: 1)')
parser.add_argument('--gpu', default='0', help='index of gpus to use')
parser.add_argument('--epochs', default=1, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=4, type=int,
                    metavar='N', help='mini-batch size (default: 200)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--lr_step', default='5', help='decreasing strategy')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='./model_fine_tune_epoch_150.hdf5', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')#default='./cub200/checkpoint_pretrain_vgg16_bn.pth.tar'
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--first_epochs', default=5, type=int, metavar='N',
                    help='number of first stage epochs to run')
parser.add_argument('--students', default='beginners', help='user type')
parser.add_argument('--maps', default='abs', help='explanation maps')



best_prec1 = 0

def main():
    global args, best_prec1
    args = parser.parse_args()

    # select gpus
    args.gpu = args.gpu.split(',')
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(args.gpu)

    # data loader
    base_path = 'G:/CUB_200_2011/CUB_200_2011'
    data_dir =base_path+'/train_test_split/train/'
    data_dir_test =base_path+'/train_test_split/test/'
    label_map = np.loadtxt(fname=base_path + '/classes.txt',dtype='str')
    label_map = label_map[:,1].tolist()
    print('using official split')
    num_classes=len(label_map)
    print("num classes:", num_classes)
    
    batch_size = 16
    input_shape = (batch_size,224,224,3)
    
    #choose limited classes to test;
    chosen_class_ids = np.array([9,25, 108, 125, 170])
    #chosen_class_ids = np.arange(0, 200)

    chosen_classes = [label_map[i] for i in chosen_class_ids]
    imgDataGen_official_split = ImageDataGenerator(preprocessing_function = preprocess_input)
    test_gen  = imgDataGen_official_split.flow_from_directory(data_dir_test,
                    target_size=(input_shape[1], input_shape[2]),
                    color_mode='rgb',
                    class_mode='categorical',
                    batch_size=batch_size,
                    shuffle=False,
                    seed=None,
                    #subset='validation',
                    interpolation='nearest',
                    classes=chosen_classes)    
    # assert callable(datasets.__dict__[args.dataset])
    # get_dataset = getattr(datasets, args.dataset)
    # num_classes = 200#datasets._NUM_CLASSES[args.dataset]
    # train_loader, val_loader = get_dataset(
    #     batch_size=args.batch_size, num_workers=args.workers)

    # create model
    vgg = VGG16(weights='imagenet',include_top = False,input_shape=(224,224,3))#top needed to get output dimensions at each layer
    base_model = tf.keras.Model(vgg.input,vgg.layers[-2].output)
    max_pool =  MaxPool2D()(base_model.output)
    mean_fmap = GlobalAveragePooling2D()(max_pool)
    x = tf.keras.layers.Dropout(0.5)(mean_fmap)
    logits = Dense(num_classes,activation=None)(x)
    softmax = tf.keras.layers.Activation('softmax')(logits)

    model = tf.keras.Model(inputs=base_model.input, outputs= [base_model.output,logits,softmax])
    #model = tf.keras.Model(inputs=base_model.input, outputs= [softmax])
    
    model.summary()
    model.load_weights(filepath='./model_fine_tune_epoch_150.hdf5')
    model.compile(optimizer=optimizers.SGD(lr=0.01/10, momentum = 0.9), 
              loss=[categorical_crossentropy], 
              metrics=['accuracy'])
    test = False
    if test:
            
        _,_,pred_probs = model.predict(test_gen,verbose=1)
        pred_classes = np.argmax(pred_probs,1)
        #actual_classes = np.argmax(test_gen.y,1)
        actual_classes = chosen_class_ids[test_gen.classes]
        print(confusion_matrix(actual_classes,pred_classes))
        print(classification_report(actual_classes,pred_classes,digits=4))
    
    model_main = model
    #model_main = models.__dict__['vgg16_bn'](pretrained=True)
    #model_main.classifier[-1] = nn.Linear(model_main.classifier[-1].in_features, num_classes)
    #model_main = torch.nn.DataParallel(model_main, device_ids=range(len(args.gpu))).cuda()
    
    # if args.resume:
    #     if os.path.isfile(args.resume):
    #         print("=> loading checkpoint '{}'".format(args.resume))
    #         checkpoint = torch.load(args.resume)
    #         model_main.module.load_state_dict(checkpoint['state_dict_m'])
    #     else:
    #         print("=> no checkpoint found at '{}'".format(args.resume))


    # if args.students == 'beginners':
    #     all_correct_student = np.load('./cub200/all_correct_random_te.npy')
    #     all_predicted_student = np.load('./cub200/all_predicted_random_te.npy')
    #     all_gt_target_student = np.load('./cub200/all_gt_target_random_te.npy')
    # else:
    #     all_correct_student = np.load('./cub200/all_correct_alexnet_te.npy')
    #     all_predicted_student = np.load('./cub200/all_predicted_alexnet_te.npy')
    #     all_gt_target_student = np.load('./cub200/all_gt_target_alexnet_te.npy')


    # generate predicted hardness score
    #criterion = nn.CrossEntropyLoss().cuda()
    #criterion_f = nn.CrossEntropyLoss(reduce=False).cuda()
    #prec1, prec5, all_correct_te, all_predicted_te, all_entropy_te, all_class_dis_te, all_gt_target_te = validate(val_loader, model_main, criterion, criterion_f)
    
    #commented if already saved
    # all_predicted_te, all_gt_target_te, all_top2_classes = validate_ali(test_gen, model_main, chosen_class_ids)

    # all_predicted_te = all_predicted_te.astype(int)
    # all_top2_classes = all_top2_classes.astype(int)

    # np.save('./cub200/all_predicted_te_cls_vgg16.npy', all_predicted_te)
    # np.save('./cub200/all_gt_target_te_cls_vgg16.npy', all_gt_target_te)
    # np.save('./cub200/all_top2_classes_cls_vgg16.npy', all_top2_classes)

    all_predicted_teacher = np.load('./cub200/all_predicted_te_cls_vgg16.npy')
    all_gt_target_teacher = np.load('./cub200/all_gt_target_te_cls_vgg16.npy')
    all_top2_classes_teacher = np.load('./cub200/all_top2_classes_cls_vgg16.npy')
    # np.random.seed(seed = 11)
    # all_top2_classes_teacher = np.asarray([np.random.randint(0,200) for i in range(150)])

    # all_predicted_teacher = np.load('./cub200/all_predicted_te_cls_vgg16_all_classes.npy')
    # all_gt_target_teacher = np.load('./cub200/all_gt_target_te_cls_vgg16_all_classes.npy')
    # all_top2_classes_teacher = np.load('./cub200/all_top2_classes_cls_vgg16_all_classes.npy')
    
    # in order to model machine teaching, the examples we care about should be those that student network misclassified but teacher network make it
    # interested_idx = np.intersect1d(np.where(all_correct_student == 0), np.where(all_correct_teacher == 1))
    # predicted_class = all_predicted_student[interested_idx]
    # counterfactual_class = all_gt_target_student[interested_idx]
    
    #predicted_class is actually the wrong CF class for images that were correctly predicted by VGG model 
    predicted_class = all_top2_classes_teacher#all_predicted_teacher
    counterfactual_class = all_predicted_teacher#all_gt_target_teacher#all_top2_classes_teacher
    #TODO: verify top2 class and GT class can be same
    #need to set CF class as the actual predicted class
    
    # pick the interested images
    # imlist = []
    # imclass = []
    # with open('./cub200/CUB_200_2011/CUB200_gt_te.txt', 'r') as rf:
    #     for line in rf.readlines():
    #         impath, imlabel, imindex = line.strip().split()
    #         imlist.append(impath)
    #         imclass.append(imlabel)

    # picked_list = []
    # picked_class_list = []
    # for i in range(np.size(interested_idx)):
    #     picked_list.append(imlist[interested_idx[i]])
    #     picked_class_list.append(imclass[interested_idx[i]])
    ##ali: interested images are same as all test images of selected few classes
    #but we cannot have same classes for them; predicted class and targetclass must be different

    heat_map_hp = Heatmap_hp(model_main, target_layer_names=["block5_conv3"], use_cuda=True)
    heat_map_cls = Heatmap_cls(model_main, target_layer_names=["block5_conv3"], use_cuda=True)

    dis_extracted_attributes = np.load('./cub200/Dominik2003IT_dis_extracted_attributes_02.npy',allow_pickle=True)
    all_locations = np.zeros((5794, 30))
    
    #Done: need to update part locations according to the chosen class images; or extract corresponding indexes of chosen class images
    #TODO: updated the file... but need to verify if it is correct now
    with open('./cub200/CUB200_partLocs_gt_te.txt', 'r') as rf:
        for line in rf.readlines():
            locations = line.strip().split()
            for i_part in range(30):
                all_locations[int(locations[-1]), i_part] = round(float(locations[i_part]))
    picked_locations = all_locations#[interested_idx, :]



    # save cub200 hard info
    # cub200cf = './cub200/CUB200cf_gt_te.txt'
    # fl = open(cub200cf, 'w')
    # num_cf = 0
    # for ii in range(len(picked_list)):

    #     example_info = picked_list[ii] + " " + picked_class_list[ii] + " " + str(num_cf)
    #     fl.write(example_info)
    #     fl.write("\n")
    #     num_cf = num_cf + 1
    # fl.close()

    # # data loader
    # assert callable(datasets.__dict__['cub200cf'])
    # get_dataset = getattr(datasets, 'cub200cf')
    # num_classes = datasets._NUM_CLASSES['cub200cf']
    # _, val_hard_loader = get_dataset(
    #     batch_size=25, num_workers=args.workers)
    
    #ali: same as test_gen with limited chosen classes
    val_hard_loader = test_gen
    picked_list = np.asarray(val_hard_loader.filepaths)
    
    #TODO: create image list or adapt code to use generator?
    
    print(args.students)
    print(args.maps)
    remaining_mask_size_pool = np.arange(0.01, 1.0, 0.01)
    remaining_mask_size_pool = np.concatenate(([0.005], remaining_mask_size_pool))
    recall, precision,i_sample = cf_proposal_extraction(val_hard_loader, heat_map_hp, heat_map_cls,
                                                                     picked_list, dis_extracted_attributes,
                                                                     picked_locations, predicted_class,
                                                                     remaining_mask_size_pool, args.maps,chosen_class_ids,counterfactual_class)



    print(np.nanmean(recall[0:i_sample], axis=0))
    print(np.nanmean(precision[0:i_sample], axis=0))

    return recall, precision




def validate_ali(val_loader, model_main, chosen_class_ids):
    _,_,pred_probs = model_main.predict(val_loader,verbose=1)
    # pred_probs = model_main.predict(val_loader,verbose=1)
    pred_classes = np.argmax(pred_probs,1)
    #actual_classes = np.argmax(test_gen.y,1)
    #   actual_classes = val_loader.classes
    actual_classes = chosen_class_ids[val_loader.classes]

    pred_probs_copy = pred_probs.copy()
    for i in range(len(pred_probs_copy)):
        pred_probs_copy[i,pred_classes[i]] = 0
    all_top2_classes = np.argmax(pred_probs_copy,1)
    
    return pred_classes, actual_classes, all_top2_classes



def largest_indices(ary, n):
    """Returns the n largest indices from a numpy array."""
    flat = ary.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ary.shape)


def largest_indices_each_example(all_response, topK):
    topK_maxIndex = np.zeros((np.size(all_response, 0), topK), dtype=np.int16)
    topK_maxValue = np.zeros((np.size(all_response, 0), topK))
    for i in range(np.size(topK_maxIndex, 0)):
        arr = all_response[i, :]
        topK_maxIndex[i, :] = np.argsort(arr)[-topK:][::-1]
        topK_maxValue[i, :] = np.sort(arr)[-topK:][::-1]
    return topK_maxIndex, topK_maxValue


def save_predicted_hardness(train_loader, val_loader, model_ahp_trunk, model_ahp_hp):
    model_ahp_trunk.eval()
    model_ahp_hp.eval()
    # hardness_scores_tr = []
    # hardness_scores_idx_tr = []
    # for i, (input, target, index) in enumerate(train_loader):
    #     input = input.cuda()
    #     predicted_hardness_scores =  model_ahp_hp(model_ahp_trunk(input)).squeeze()
    #     scores = predicted_hardness_scores.data.cpu().numpy()
    #     hardness_scores_tr = np.concatenate((hardness_scores_tr, scores), axis=0)
    #     index = index.numpy()
    #     hardness_scores_idx_tr = np.concatenate((hardness_scores_idx_tr, index), axis=0)

    hardness_scores_val = []
    hardness_scores_idx_val = []
    for i, (input, target, index) in enumerate(val_loader):
        input = input.cuda()
        trunk_output = model_ahp_trunk(input)
        predicted_hardness_scores, _ = model_ahp_hp(trunk_output)
        scores = predicted_hardness_scores.data.cpu().numpy().squeeze()
        hardness_scores_val = np.concatenate((hardness_scores_val, scores), axis=0)
        index = index.numpy()
        hardness_scores_idx_val = np.concatenate((hardness_scores_idx_val, index), axis=0)

    return hardness_scores_val, hardness_scores_idx_val


# def save_checkpoint(state, filename='checkpoint_res.pth.tar'):
#     torch.save(state, filename)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res



class FeatureExtractor_hp():
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        #this function is simply getting the last conv layer output before and after maxpooling;
        #outputs =  ?*512*14*14
        #x =        ?*512*7*7
        for name, module in self.model._modules['module']._modules['features']._modules.items():
            x = module(x)  # forward one layer each time
            if name in self.target_layers:  # store the gradient of target layer
                x.register_hook(self.save_gradient)
                outputs += [x]  # after last feature map, nn.MaxPool2d(kernel_size=2, stride=2)] follows
        return outputs, x


class FeatureExtractor_cls():
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules['module']._modules['features']._modules.items():
            x = module(x)  # forward one layer each time
            if name in self.target_layers:  # store the gradient of target layer
                x.register_hook(self.save_gradient)
                outputs += [x]  # after last feature map, nn.MaxPool2d(kernel_size=2, stride=2)] follows
        return outputs, x

class ModelOutputs_hp():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """
    def __init__(self, model, target_layers):
        self.model = model
        self.feature_extractor = FeatureExtractor_hp(self.model, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        #target_activations, output = self.feature_extractor(x)
        #target_activations =    ?*512*14*14
        #output =                ?*512*7*7
        
        #Flatten
        #output = output.view(output.size(0), -1)
        
        #classifier output
        #output = ?*200 logits
        #output = self.model._modules['module'].classifier(output)  # travel many fc layers
        
        #softmax
        #confidence_score = F.softmax(output, dim=1)
        
        #max confidence score of each input image
        #confidence_score = torch.max(confidence_score, dim=1)[0]
        
        #ali: get all outputs
        target_activations,logits,softmax = self.model(x)
        confidence_score = np.max(softmax, axis=1)#[0]
        
        return target_activations, confidence_score


class ModelOutputs_cls():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """
    def __init__(self, model, target_layers):
        self.model = model
        self.feature_extractor = FeatureExtractor_cls(self.model, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations, output = self.feature_extractor(x)
        output = output.view(output.size(0), -1)
        output = self.model._modules['module'].classifier(output)  # travel many fc layers
        return target_activations, output



def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cam = np.uint8(255 * cam)
    return cam

def show_segment_on_image(img, mask, com_attributes_positions=None, all_attributes_positions=None, is_cls=True):
    img = np.float32(img)
    img_dark = np.copy(img)
    # if is_cls == False:
    #     threshold = np.sort(mask.flatten())[-int(0.05*224*224)]
    #     mask[mask < threshold] = 0
    #     mask[mask > 0] = 1
    mask = np.concatenate((mask[:, :, np.newaxis], mask[:, :, np.newaxis], mask[:, :, np.newaxis]), axis=2)
    img = np.uint8(255 * mask * img)
    if is_cls == False:
        if np.sum(com_attributes_positions*mask[:,:,0]) > 0:
            x, y = np.where(com_attributes_positions*mask[:,:,0] == 1)
            for i in range(np.size(x)):
                cv2.circle(img, (y[i], x[i]), 5, (0,255,0),-1)

            x, y = np.where((all_attributes_positions - com_attributes_positions) * mask[:, :, 0] == 1)
            for i in range(np.size(x)):
                cv2.circle(img, (y[i], x[i]), 5, (0,0,255),-1)

    # using dark images

    img_dark = img_dark * 0.4
    img_dark = np.uint8(255 * img_dark)
    img_dark[mask > 0] = img[mask > 0]
    img = img_dark

    return img


class Heatmap_hp:
    def __init__(self, model, target_layer_names, use_cuda):
        self.model = model
        #self.model.eval()
        #self.cuda = use_cuda
        #if self.cuda:
        #    self.model = model.cuda()

        #self.extractor = ModelOutputs_hp(self.model, target_layer_names)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input):

        #features = ?*512*14*14; output = ?*1 (max softmax of each image)
        #features, output = self.extractor(input)
        
        with tf.GradientTape() as tape:

            features,logits,softmax = self.model(input)
            output = tf.reduce_max(softmax, axis=1)#[0]
        grads_val = tape.gradient(output, features)

 
        #compute gradients of features with respect to output
        #grads_val = torch.autograd.grad(output, features[0], grad_outputs=torch.ones_like(output),
        #                                create_graph=True)
        #grads_val = grads_val[0].squeeze()
        #grads_val = grads_val.cpu().data.numpy()
        #grads_val = ?*512*14*14
        
        grads_val = grads_val.numpy()
        
        mask_positive = np.copy(grads_val)
        mask_positive[mask_positive < 0.0] = 0.0
        mask_positive = mask_positive.squeeze()

        target = features#[-1]#same as features[0] if length of features is 1
        target = target.numpy()

        cam_positive = target * mask_positive
        cam_positive = np.sum(cam_positive, axis=3)

        return cam_positive



class Heatmap_cls:
    def __init__(self, model, target_layer_names, use_cuda):
        self.model = model
        #self.model.eval()
        #self.cuda = use_cuda
        #if self.cuda:
        #    self.model = model.cuda()

        self.extractor = ModelOutputs_cls(self.model, target_layer_names)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, CounterClass, TargetClass):
        #features = ?*512*14*14, output = logits classifier output = ?*200
        #features, output = self.extractor(input)
        CounterClass = tf.keras.utils.to_categorical(CounterClass,200)
        with tf.GradientTape() as tape:

            features,logits,softmax = self.model(input)
            #output = tf.reduce_max(softmax, axis=1)#[0]
            output = tf.reduce_sum(softmax*CounterClass,axis=1)
        grads_val = tape.gradient(output, features)

        target = features#[-1]
        target = target.numpy()#cpu().data.numpy()

        classifier_heatmaps = np.zeros((input.shape[0], np.size(target, 2), np.size(target, 2), 2))
        # one_hot = np.zeros((output.shape[0], output.size()[-1]), dtype=np.float32)
        # one_hot[np.arange(output.shape[0]), CounterClass] = 1
        # one_hot = Variable(torch.from_numpy(one_hot), requires_grad=True)
        # one_hot = torch.sum(one_hot.cuda() * output, dim=1)
        
        #compute gradients of features with respect to predicted class logits
        #grads_val = torch.autograd.grad(one_hot, features, grad_outputs=torch.ones_like(one_hot),
        #                                create_graph=True)
        #grads_val = grads_val[0].squeeze()
        #grads_val = grads_val.cpu().data.numpy().squeeze()
        
        grads_val = grads_val.numpy()
        cam_positive = target * grads_val
        cam_positive = np.sum(cam_positive, axis=3)
        classifier_heatmaps[:, :, :, 0] = cam_positive


        TargetClass = tf.keras.utils.to_categorical(TargetClass,200)
        with tf.GradientTape() as tape:

            features,logits,softmax = self.model(input)
            #output = tf.reduce_max(softmax, axis=1)#[0]
            output = tf.reduce_sum(softmax*TargetClass,axis=1)
        grads_val = tape.gradient(output, features)
        # one_hot = np.zeros((output.shape[0], output.size()[-1]), dtype=np.float32)
        # one_hot[np.arange(output.shape[0]), TargetClass] = 1
        # one_hot = Variable(torch.from_numpy(one_hot), requires_grad=True)
        # one_hot = torch.sum(one_hot.cuda() * output, dim=1)
        
        #compute gradients of features with respect to CF/target class logits
        #grads_val = torch.autograd.grad(one_hot, features, grad_outputs=torch.ones_like(one_hot),
        #                                create_graph=True)
        #grads_val = grads_val[0].squeeze()
        #grads_val = grads_val.cpu().data.numpy().squeeze()
        
        grads_val = grads_val.numpy()

        cam_positive = target * grads_val
        cam_positive = np.sum(cam_positive, axis=3)
        classifier_heatmaps[:, :, :, 1] = cam_positive


        return classifier_heatmaps

def image_to_uint_255(image):
    """
    Convert float images to int 0-255 images.

    Args:
        image (numpy.ndarray): Input image. Can be either [0, 255], [0, 1], [-1, 1]

    Returns:
        numpy.ndarray:
    """
    if image.dtype == np.uint8:
        return image

    if image.min() < 0:
        image = (image + 1.0) / 2.0

    return (image * 255).astype("uint8")

def heatmap_area_display(
    heatmap, original_image, colormap=cv2.COLORMAP_VIRIDIS, image_weight=0.7
):
    """
    Apply a heatmap (as an np.ndarray) on top of an original image.

    Args:
        heatmap (numpy.ndarray): Array corresponding to the heatmap
        original_image (numpy.ndarray): Image on which we apply the heatmap
        colormap (int): OpenCV Colormap to use for heatmap visualization
        image_weight (float): An optional `float` value in range [0,1] indicating the weight of
            the input image to be overlaying the calculated attribution maps. Defaults to `0.7`

    Returns:
        np.ndarray: Original image with heatmap applied
    """
    #heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))

    image = image_to_uint_255(original_image)

    heatmap = (heatmap - np.min(heatmap)) / (heatmap.max() - heatmap.min())
    
    original_heatmap = (heatmap * 255).astype("uint8")
    
    #heatmap = cv2.applyColorMap(
    #    cv2.cvtColor((heatmap * 255).astype("uint8"), cv2.COLOR_GRAY2BGR), colormap
    #)
    
    ## ali - convert heatmap to mask
    heatmap_mask = np.ones(heatmap.shape)*0.6
    heatmap_mask[heatmap>0.6] = 1
    
    #plot red outline around the mask
    ret, thresh = cv2.threshold((heatmap_mask * 255).astype("uint8"), 200, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    
    #in case 2d MNIST image
    if len(image.shape) ==2:
        #convert to 3d image
        image = np.stack((image,)*3, axis=-1)
        
        #dont dim background of MNIST images
        #output = image * heatmap_mask[:,:,np.newaxis]
        output = image

        #converted = cv2.cvtColor((heatmap * 255).astype("uint8"), cv2.COLOR_GRAY2BGR)
        cv2.drawContours(output, contours, -1, (255,0,0), 3)
        
    else:
        output = image * heatmap_mask[:,:,np.newaxis]
        cv2.drawContours(output, contours, -1, (255,0,0), 3)

    

    # output = cv2.addWeighted(
    #     cv2.cvtColor(image, cv2.COLOR_RGB2BGR), image_weight, heatmap, 1, 0
    # )

    return output.astype("uint8")# cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

#for VGG preprocessing only
def restore_original_image_from_array(x, data_format='channels_last'):
    mean = [103.939, 116.779, 123.68]

    # Zero-center by mean pixel
    if data_format == 'channels_first':
        if x.ndim == 3:
            x[0, :, :] += mean[0]
            x[1, :, :] += mean[1]
            x[2, :, :] += mean[2]
        else:
            x[:, 0, :, :] += mean[0]
            x[:, 1, :, :] += mean[1]
            x[:, 2, :, :] += mean[2]
    else:
        x[..., 0] += mean[0]
        x[..., 1] += mean[1]
        x[..., 2] += mean[2]

    if data_format == 'channels_first':
        # 'BGR'->'RGB'
        if x.ndim == 3:
            x = x[::-1, ...]
        else:
            x = x[:, ::-1, ...]
    else:
        # 'BGR'->'RGB'
        x = x[..., ::-1]

    return x

def cf_proposal_extraction(val_loader, heat_map_hp, heat_map_cls, imglist, dis_extracted_attributes, part_Locs, predicted_class, remaining_mask_size_pool, maps, chosen_class_ids,counterfactual_class):


    recall = np.zeros((len(imglist), np.size(remaining_mask_size_pool)))
    precision = np.zeros((len(imglist), np.size(remaining_mask_size_pool)))
    total_times = []
    i_sample = 0
    #for i, (input, target, index) in enumerate(val_loader):
    index = np.zeros(val_loader.batch_size,np.int32)
    batches = np.floor(len(val_loader.filenames)/val_loader.batch_size)
    miclassifications = 0
    for i, (input, target) in enumerate(val_loader):
        #input = input.cuda()
        #TODO: target class needs to be the actual predicted class, not the GT class
        #predicted class in this context is the CF class
       
        #need to manually break loop when using generators 
        if i == batches:
           break
        print('processing batch', i, ' of ', batches)
        gt_class = chosen_class_ids[np.argmax(target,1)]

        #target = [counterfactual_class[i] for i in index]#chosen_class_ids[val_loader.classes]
        index = np.arange(i*val_loader.batch_size,i*val_loader.batch_size+len(target))
        target = counterfactual_class[index]
        
        start = timeit.default_timer()
        easiness_heatmaps_set = heat_map_hp(input)
        easiness_mask_set = np.copy(easiness_heatmaps_set)
        easiness_mask_set[easiness_mask_set > 0] = 1
        classifier_heatmaps_set = heat_map_cls(input, predicted_class[index], target)
        classifier_heatmaps_set[classifier_heatmaps_set < 0] = 1e-7
        counter_class_heatmaps_set = classifier_heatmaps_set[:, :, :, 0]
        prediction_class_heatmaps_set = classifier_heatmaps_set[:, :, :, 1]
        stop = timeit.default_timer()
        print('Time:', stop - start)
        total_times.append(stop - start)
        
        #target = np.argmax(target,axis=1)
        for i_batch in range(index.shape[0]):
            if gt_class[i_batch]!=target[i_batch]:
                print("skipping misclassification")
                miclassifications = miclassifications+1
                continue
            easiness_heatmaps = easiness_heatmaps_set[i_batch, :, :].squeeze()
            easiness_mask = easiness_mask_set[i_batch, :, :].squeeze()
            counter_class_heatmaps = counter_class_heatmaps_set[i_batch, :, :].squeeze()
            prediction_class_heatmaps = prediction_class_heatmaps_set[i_batch, :, :].squeeze()

            img = cv2.imread(imglist[index[i_batch]])
            img_X_max = np.size(img, axis=0)
            img_Y_max = np.size(img, axis=1)
            part_Locs_example = part_Locs[index[i_batch], :]
            part_Locs_example = np.concatenate((np.reshape(part_Locs_example[0::2], (-1, 1)), np.reshape(part_Locs_example[1::2], (-1, 1))), axis=1)
            part_Locs_example[:, 0] = 224.0 * part_Locs_example[:, 0] / img_Y_max
            part_Locs_example[:, 1] = 224.0 * part_Locs_example[:, 1] / img_X_max
            part_Locs_example = np.round(part_Locs_example)
            part_Locs_example = part_Locs_example.astype(int)

            for i_remain in range(np.size(remaining_mask_size_pool)):
                remaining_mask_size = remaining_mask_size_pool[i_remain]
                #maps = 'a'
                if maps == 'a':
                    cf_heatmap = prediction_class_heatmaps
                elif maps == 'ab':
                    cf_heatmap = (np.amax(counter_class_heatmaps) - counter_class_heatmaps) * prediction_class_heatmaps
                else:
                    cf_heatmap = easiness_heatmaps * (np.amax(counter_class_heatmaps) - counter_class_heatmaps) * prediction_class_heatmaps
                
                cf_heatmap = cv2.resize(cf_heatmap, (224, 224))
                cf_heatmap_non_threshold=cf_heatmap.copy()
                threshold = np.sort(cf_heatmap.flatten())[int(-remaining_mask_size * 224 * 224)]
                cf_heatmap[cf_heatmap > threshold] = 1
                cf_heatmap[cf_heatmap < 1] = 0
                
                show_images = True
                if show_images:

    
                    #cam = show_cam_on_image(input[0], cf_heatmap)
                    input_post_process = restore_original_image_from_array(input[i_batch].squeeze().copy())
                    input_post_process = input_post_process.astype('uint8')
                    heatmap_display = heatmap_area_display(cf_heatmap_non_threshold, input_post_process)
                    heatmap_display_thresholded = heatmap_area_display(cf_heatmap, input_post_process)
    
                    # plt.imshow(cf_heatmap_non_threshold),plt.show()
                    # plt.imshow(cf_heatmap),plt.show()
                    plt.imshow(heatmap_display),plt.axis('off'), plt.title('SCOUT'), plt.show()
                    plt.imshow(heatmap_display_thresholded),plt.show()
                    break
                
                all_attributes_positions = np.zeros((224, 224))
                dis_attributes_positions = np.zeros((224, 224))

                dis_attributes = dis_extracted_attributes[predicted_class[index[i_batch]], target[i_batch]]

                # if len(dis_attributes) > 1:
                #     print('debug')

                if isinstance(dis_attributes, int):
                    dis_attributes = [dis_attributes]

                if len(dis_attributes) < 1:
                    recall[i_sample, i_remain] = float('NaN')
                    precision[i_sample, i_remain] = float('NaN')
                    continue

                dis_attributes = np.array(dis_attributes)
                # print(dis_attributes)

                part_Locs_example_copy = np.copy(part_Locs_example)
                part_Locs_example_copy = part_Locs_example_copy[~np.all(part_Locs_example_copy == 0, axis=1)]
                all_attributes_positions[part_Locs_example_copy[:, 1], part_Locs_example_copy[:, 0]] = 1

                dis_attributes_positions[part_Locs_example[dis_attributes, 1], part_Locs_example[dis_attributes, 0]] = 1
                dis_attributes_positions[0, 0] = 0
                # print(np.sum(dis_attributes_positions))


                cur_recall = np.sum(cf_heatmap * dis_attributes_positions) / np.sum(dis_attributes_positions)

                cur_precision = np.sum(cf_heatmap * dis_attributes_positions) / np.sum(cf_heatmap * all_attributes_positions)

                recall[i_sample, i_remain] = cur_recall
                precision[i_sample, i_remain] = cur_precision
            i_sample = i_sample + 1
    total_times = np.array(total_times)
    np.save('./cub200/ss_vgg_total_times.npy', total_times)
    print("total misclassifications: ", miclassifications)
    #return np.nanmean(recall[0:i_sample], axis=0), np.nanmean(precision[0:i_sample], axis=0)
    return recall, precision, i_sample


if __name__ == '__main__':
    recall, precision = main()



