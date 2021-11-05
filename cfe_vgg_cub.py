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
from tf_explain_modified.core.grad_cam import GradCAM


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
    global label_map

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
    
    top_filters = base_model.output_shape[3] # flters in top conv layer (512 for VGG)
    fmatrix = tf.keras.layers.Input(shape=(top_filters))
    
    max_pool =  MaxPool2D()(base_model.output)
    mean_fmap = GlobalAveragePooling2D()(max_pool)
    
    modified_fmap = mean_fmap*fmatrix
    pre_softmax = Dense(num_classes,activation=None)(modified_fmap)
    out = tf.keras.layers.Activation('softmax')(pre_softmax)
    model = tf.keras.Model(inputs=[base_model.input, fmatrix], outputs= [out,base_model.output, mean_fmap, modified_fmap,pre_softmax],name='base_model')

    default_fmatrix = tf.ones((test_gen.batch_size,base_model.output.shape[3]))


    #build CF matrix generator and combined model:
    num_filters = model.output[1].shape[3]
    model.trainable = False
    x =  MaxPool2D()(base_model.output)
    mean_fmap = GlobalAveragePooling2D()(x)
    x = Dense(num_filters,activation='sigmoid')(mean_fmap)#kernel_regularizer='l1' #,activity_regularizer='l1'
    
    thresh=0.5
    PP_filter_matrix = tf.keras.layers.ThresholdedReLU(theta=thresh)(x)
    counterfactual_generator = tf.keras.Model(inputs=base_model.input, outputs= [PP_filter_matrix],name='counterfactual_model')
    counterfactual_generator.summary()
    
    img = tf.keras.Input(shape=model.input_shape[0][1:4])
    fmatrix = counterfactual_generator(img)
    alter_prediction,fmaps,mean_fmap, modified_mean_fmap_activations,pre_softmax = model([img,fmatrix])
    
    combined = tf.keras.Model(inputs=img, outputs=[alter_prediction,fmatrix,fmaps,mean_fmap,modified_mean_fmap_activations,pre_softmax])

    # x = tf.keras.layers.Dropout(0.5)(mean_fmap)
    # logits = Dense(num_classes,activation=None)(x)
    # softmax = tf.keras.layers.Activation('softmax')(logits)
    # model = tf.keras.Model(inputs=base_model.input, outputs= [base_model.output,logits,softmax])
    
    model.summary()
    model.load_weights(filepath='./model_fine_tune_epoch_150.hdf5')
    # model.compile(optimizer=optimizers.SGD(lr=0.01/10, momentum = 0.9), 
    #           loss=[categorical_crossentropy], 
    #           metrics=['accuracy'])
    test = False
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    test_acc_metric = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')
    test_loss_metric = tf.keras.metrics.Mean(name='test_loss')    
    if test:
            
        # pred_probs,_,_,_,_ = model.predict(test_gen,verbose=1)
        # pred_classes = np.argmax(pred_probs,1)
        # #actual_classes = np.argmax(test_gen.y,1)
        # actual_classes = chosen_class_ids[test_gen.classes]
        # print(confusion_matrix(actual_classes,pred_classes))
        # print(classification_report(actual_classes,pred_classes,digits=4))
        print('Testing...')
    
        #model.load_weights(weights_path)
        batches=math.ceil(test_gen.n/test_gen.batch_size)
        default_fmatrix_test = tf.ones((test_gen.batch_size,base_model.output.shape[3]))
    
        #for step,(x_batch_train, y_batch_train) in enumerate(dataset):
        with tqdm(total=batches,file=sys.stdout,mininterval=10000) as progBar:
            #pass
            for step in range(batches):
              x_batch_test, y_batch_test = next(test_gen)
              y_batch_test = chosen_class_ids[np.argmax(y_batch_test,1)]
              y_batch_test = to_categorical(y_batch_test,200)
              default_fmatrix_test = tf.ones((len(x_batch_test),base_model.output.shape[3]))
              
              predictions, fmaps,_,_,_ = model([x_batch_test,default_fmatrix_test], training=False)              
              loss_value = loss_fn(y_batch_test, predictions)
              test_loss_metric(loss_value)
              test_acc_metric(y_batch_test, predictions)

              progBar.set_postfix(loss=test_loss_metric.result().numpy(), acc=test_acc_metric.result().numpy(), refresh=False)
              progBar.update()
            #progBar.refresh()    
          
             # Display metrics at the end of each epoch.
            test_acc = test_acc_metric.result()
            test_loss = test_loss_metric.result()
        print('\nTest loss:', test_loss.numpy())
        print('Test accuracy:', test_acc.numpy())         
    
    model_main = model
    CF_model = combined
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
    recall, precision, i_sample = cf_proposal_extraction(val_hard_loader, model_main,CF_model,picked_list, dis_extracted_attributes,
                                                                     picked_locations, predicted_class,
                                                                     remaining_mask_size_pool, args.maps,chosen_class_ids,counterfactual_class)



    print(np.nanmean(recall[0:i_sample], axis=0))
    print(np.nanmean(precision[0:i_sample], axis=0))
    return recall, precision



def validate_ali(val_loader, model_main, chosen_class_ids):
    _,_,pred_probs = model_main.predict(val_loader,verbose=1)
    pred_classes = np.argmax(pred_probs,1)
    #actual_classes = np.argmax(test_gen.y,1)
    #   actual_classes = val_loader.classes
    actual_classes = chosen_class_ids[val_loader.classes]

    pred_probs_copy = pred_probs.copy()
    for i in range(len(pred_probs_copy)):
        pred_probs_copy[i,pred_classes[i]] = 0
    all_top2_classes = np.argmax(pred_probs_copy,1)
    
    return pred_classes, actual_classes, all_top2_classes





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


def cf_proposal_extraction(val_loader,model_main, CF_model,  imglist, dis_extracted_attributes, part_Locs, predicted_class, remaining_mask_size_pool, maps, chosen_class_ids,counterfactual_class):

    recall = np.zeros((len(imglist), np.size(remaining_mask_size_pool)))
    precision = np.zeros((len(imglist), np.size(remaining_mask_size_pool)))
    total_times = []
    i_sample = 0
    #for i, (input, target, index) in enumerate(val_loader):
    index = np.zeros(val_loader.batch_size,np.int32)
    batches = np.floor(len(val_loader.filenames)/val_loader.batch_size)

    explainer = GradCAM()
    show_images = False
    current_class=''
    miclassifications = 0
    for i, (input, target) in enumerate(val_loader):
        #input = input.cuda()
        #TODO: target class needs to be the actual predicted class, not the GT class
        #predicted class in this context is the CF class
       
        #need to manually break loop when using generators 
        if i == batches:
           break
        print('processing batch', i, ' of ', batches)
        #target = [counterfactual_class[i] for i in index]#chosen_class_ids[val_loader.classes]
        gt_class = chosen_class_ids[np.argmax(target,1)]
        # gt_class = to_categorical(gt_class,200)
        
        index = np.arange(i*val_loader.batch_size,i*val_loader.batch_size+len(target))
        target = counterfactual_class[index]

        default_fmatrix = tf.ones((len(input),model_main.output[1].shape[3]))
        

        # start = timeit.default_timer()
        # classifier_heatmaps_set = heat_map_cls(input, predicted_class[index], target)
        # classifier_heatmaps_set[classifier_heatmaps_set < 0] = 1e-7
        # counter_class_heatmaps_set = classifier_heatmaps_set[:, :, :, 0]
        # prediction_class_heatmaps_set = classifier_heatmaps_set[:, :, :, 1]
        # stop = timeit.default_timer()
        # print('Time:', stop - start)
        # total_times.append(stop - start)
        
        #target = np.argmax(target,axis=1)
        for i_batch in range(index.shape[0]):
            
            #skip wrong predictions
            #TODO: set recall and precision array initializes correctly be pre-checking misclassifications
            if gt_class[i_batch]!=target[i_batch]:
                print("skipping misclassification")
                miclassifications=miclassifications+1
                continue
            
            if current_class != str(label_map[target[i_batch]]):
                current_class = str(label_map[target[i_batch]])
                print("Loading CF model for ",str(label_map[target[i_batch]])+'_alter_class')
                CF_model.load_weights(filepath='./counterfactual_combined_model_fixed_'+str(label_map[target[i_batch]])+'_alter_class.hdf5')
            
            #get gradCAM heatmap of thresholded CF prediction
            if show_images:
                pred_probs,fmaps,mean_fmaps,_ ,pre_softmax= model_main([np.expand_dims(input[i_batch],0),np.expand_dims(default_fmatrix[0],0)], training=False)#with eager
                #pred_probs,fmaps_x1,fmaps_x2,target_1,target_2,raw_map,forward_1 = model(np.expand_dims(x_batch_test[img_ind],0),y_batch_test[img_ind])#with eager

                print('predicted: ',label_map[np.argmax(pred_probs)], ' with prob: ',np.max(pred_probs)*100,'%')
                print ('actual: ', label_map[gt_class[i_batch]], ' with prob: ',pred_probs[0][gt_class[i_batch]].numpy()*100,'%')
                
                input_post_process = restore_original_image_from_array(input[i_batch].squeeze().copy())
                input_post_process = input_post_process.astype('uint8')
                output_orig,_ = explainer.explain((np.expand_dims(input[i_batch],0),None),model_main,np.argmax(pred_probs),image_nopreprocessed=np.expand_dims(input_post_process,0),fmatrix=default_fmatrix)
                
                plt.imshow(output_orig), plt.axis('off'), plt.title('original prediction')
                plt.show()
            # pred_probs,fmaps,mean_fmaps,_ ,pre_softmax= model_main([np.expand_dims(input[i_batch],0),np.expand_dims(default_fmatrix[0],0)], training=False)#with eager
            # output_cf,cams = explainer.explain((np.expand_dims(input[i_batch],0),None),model_main,np.argmax(pred_probs),image_nopreprocessed=np.expand_dims(input[i_batch],0),fmatrix=default_fmatrix)

            
            alter_prediction,fmatrix,fmaps, mean_fmap, modified_mean_fmap_activations,alter_pre_softmax = CF_model(np.expand_dims(input[i_batch],0))
            filters_off = fmatrix
            t_fmatrix = filters_off.numpy()
            for i in tf.where(filters_off>0):
                t_fmatrix[tuple(i)]=1.0
            t_fmatrix = tf.convert_to_tensor(t_fmatrix)            
            
            alter_class = target[i_batch]
            if show_images:
                alter_probs, c_fmaps, c_mean_fmap, c_modified_mean_fmap_activations,alter_pre_softmax = model_main([np.expand_dims(input_post_process,0),t_fmatrix])#with eager

                print('\nthresholded counterfactual')
                print( 'gt class: ',label_map[gt_class[i_batch]], '  prob: ',alter_probs[0][gt_class[i_batch]].numpy()*100,'%')
                print( 'alter class: ',label_map[alter_class], '  prob: ',alter_probs[0][alter_class].numpy()*100,'%')
            
            
                plt.plot(c_modified_mean_fmap_activations[0]),plt.ylim([0, np.max(c_mean_fmap)+1]), plt.title('thresh_mean_fmap_activations'), plt.show()
            input_post_process = restore_original_image_from_array(input[i_batch].squeeze().copy())
            input_post_process = input_post_process.astype('uint8')
            output_cf,cams = explainer.explain((np.expand_dims(input[i_batch],0),None),model_main,alter_class,image_nopreprocessed=np.expand_dims(input_post_process,0),fmatrix=t_fmatrix,image_weight=0.7)#np.argmin(y_batch_test[img_ind])

            if show_images:
                
                plt.imshow(output_cf), plt.axis('off'), plt.title('thresh prediction')
                plt.show()
                plt.imshow(np.squeeze(input_post_process)), plt.axis('off'), plt.title('original image')
                plt.show()
    
            
                fig, axs = plt.subplots(2, 2,figsize=(15,10))
                axs[0, 0].imshow(output_orig), axs[0, 0].axis('off'), axs[0, 0].set_title('original prediction')
                axs[0, 1].imshow(output_cf), axs[0, 1].axis('off'), axs[0, 1].set_title('modified prediction')
                axs[1, 0].plot(c_mean_fmap[0]),axs[1,0].set_ylim([0, np.max(c_mean_fmap)+1]), axs[1, 0].set_title('mean_fmaps')
                axs[1, 1].plot(c_modified_mean_fmap_activations[0]),axs[1,1].set_ylim([0, np.max(c_mean_fmap)+1]), axs[1, 1].set_title('modified_mean_fmap_activations')
                    
                plt.show()


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


                cf_heatmap = cams[0].numpy()#easiness_heatmaps * (np.amax(counter_class_heatmaps) - counter_class_heatmaps) * prediction_class_heatmaps


                cf_heatmap = cv2.resize(cf_heatmap, (224, 224))
                cf_heatmap_non_threshold=cf_heatmap.copy()
                threshold = np.sort(cf_heatmap.flatten())[int(-remaining_mask_size * 224 * 224)]
                cf_heatmap[cf_heatmap > threshold] = 1
                cf_heatmap[cf_heatmap < 1] = 0

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
                
                if True:
                    print(cur_recall)
                    print(cur_precision)
                    plt.plot(part_Locs_example_copy[:,0],part_Locs_example_copy[:,1],'g+',color='red')
                    plt.imshow(np.squeeze(input_post_process)), plt.axis('on'), plt.title('original image'), plt.show()
                    plt.plot(part_Locs_example_copy[:,0],part_Locs_example_copy[:,1],'g+',color='red')
                    plt.imshow(np.squeeze(cf_heatmap)), plt.axis('on'), plt.title('cf_heatmap'), plt.show()
    
                    plt.plot(part_Locs_example_copy[dis_attributes,0],part_Locs_example_copy[dis_attributes,1],'g+',color='red')
                    plt.imshow(np.squeeze(input_post_process)), plt.axis('on'), plt.title('original image'), plt.show()
    
                    plt.imshow(dis_attributes_positions), plt.axis('off'), plt.title('dis_attributes_positions'), plt.show()
                    plt.imshow(all_attributes_positions), plt.axis('off'), plt.title('all_attributes_positions'), plt.show()
                    
                    heatmap_display = heatmap_area_display(cf_heatmap_non_threshold, input_post_process)
                    heatmap_display_thresholded = heatmap_area_display(cf_heatmap, input_post_process)
    
                    # plt.imshow(cf_heatmap_non_threshold),plt.show()
                    # plt.imshow(cf_heatmap),plt.show()
                    plt.imshow(heatmap_display),plt.axis('off'), plt.title('Proposed'), plt.show()
                    plt.imshow(heatmap_display_thresholded),plt.show()

                    break
            i_sample = i_sample + 1
    total_times = np.array(total_times)
    np.save('./cub200/ss_vgg_total_times.npy', total_times)
    print("total misclassifications: ", miclassifications)
    #return np.nanmean(recall[0:i_sample], axis=0), np.nanmean(precision[0:i_sample], axis=0)
    return recall, precision, i_sample



if __name__ == '__main__':
    recall, precision = main()



