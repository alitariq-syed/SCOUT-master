import numpy as np


image_list_path = './cub200/CUB_200_2011/images.txt'
class_list_path = './cub200/CUB_200_2011/image_class_labels.txt'
tr_te_split_list_path = './cub200/CUB_200_2011/train_test_split.txt'
part_locs_list_path = './cub200/CUB_200_2011/parts/part_locs.txt'


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input

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
    
imlist = []
with open(image_list_path, 'r') as rf:
    for line in rf.readlines():
        imindex, impath = line.strip().split()
        imlist.append(impath)

classlist = []
with open(class_list_path, 'r') as rf:
    for line in rf.readlines():
        imindex, label = line.strip().split()
        classlist.append(str(int(label)-1))

tr_te_list = []
with open(tr_te_split_list_path, 'r') as rf:
    for line in rf.readlines():
        imindex, tr_te = line.strip().split()
        tr_te_list.append(tr_te)

part_id_list = []
part_locsX_list = []
part_locsY_list = []
with open(part_locs_list_path, 'r') as rf:
    for line in rf.readlines():
        imindex, part_idx, x_cord, y_cord, uncertainty = line.strip().split()
        part_id_list.append(part_idx)
        part_locsX_list.append(x_cord)
        part_locsY_list.append(y_cord)

save_partLocs_test_list = './cub200/CUB200_partLocs_gt_te.txt'

# save part Locations, <part1 x> <part1 y> <part2 x> <part2 y>,...,<part15 x> <part15 y>, <index>

#TODO: ali - find corresponding index of file in test_gen and imlist
num_te = 0
fl = open(save_partLocs_test_list, 'w')
for i in range(len(test_gen.filenames)):
    #if tr_te_list[i] == '0':
    print(i)
    filename = test_gen.filenames[i].split('\\')[1]
    #find actual index in imglist where this file is located
    for j in range(len(imlist)):
        file_imglist = imlist[j].split('/')[1]
        if filename==file_imglist:
            actual_index = j
            break
    example_info = ""
    for i_part in range(15):
        if int(part_id_list[j * 15 + i_part]) != i_part + 1:
            print('error')
        example_info = example_info + part_locsX_list[j * 15 + i_part] + " " + part_locsY_list[j * 15 + i_part] + " "
    example_info = example_info + " " + str(num_te)
    fl.write(example_info)
    fl.write("\n")
    num_te = num_te + 1
fl.close()
