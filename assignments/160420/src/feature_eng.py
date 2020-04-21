import setGPU

import tensorflow as tf

import os, csv, shutil
import cv2 as cv
import numpy as np
from random import shuffle

"""
read data raw entries from csv, create classes path and copy the data into different classes
@csv_file_path: csv file path
@src_data_path: path where all the images present
@classes_data_path: gets created with different classes in it

Createa a path "../data/classes/XXX" where XXX are class names
"""
def read_csv_file(csv_file_path, src_data_path):
    """ path where it splits data as classes based on labels and store the images in different classes """
    classes_data_path = '../data/classes'
    if not os.path.exists(classes_data_path):
        os.makedirs(classes_data_path)
    paths = []

    with open(csv_file_path) as file:
        reader = csv.reader(file)
        try:
            for row in reader:
                if row[0] == 'IMAGE_FILENAME' or row[1] == 'LABEL':
                    continue

                """ create respective paths to separate the data into different classes """
                src_file_path = os.path.join(src_data_path, row[0])
                dst_path = os.path.join(classes_data_path, row[1].strip())
                if not os.path.exists(dst_path):
                    os.makedirs(dst_path)
                    paths.append(dst_path)
                dst_file_path = os.path.join(dst_path, row[0])

                shutil.copy(src_file_path, dst_file_path)

        except csv.Error as e:
            sys.exit('line {}, {}'.format(reader.line_num, e))

    return paths

"""
Separete noise from sharp image and return sharp image channel
@file_path: frame path
"""
def get_single_channel_image(file_path):
    img = cv.imread(file_path)
    b, g, r = cv.split(img)

    """
    file_name = 'dump_file.jpg' #create temp file name to get the 3 channel image from a single channel image
    cv.imwrite(file_name, img)
    img = cv.imread(file_name)
    os.remove(file_name)"""

    return b

"""
Sepearte the single channle image form original noisy image and dumps in the given path
@data_path: path where all the augmented or oringal images present
@single_channel_path: path where all single channel images will be stored
"""
def extract_single_channel_image(data_path, single_channel_path):
    files = os.listdir(data_path)
    for file in files:
        file_path = os.path.join(data_path, file)

        img = get_single_channel_image(file_path)
        base, ext = file.split('.')
        file_name = base+'_SC'+'.'+ext
        sc_file_path = os.path.join(single_channel_path, file_name)
        cv.imwrite(sc_file_path, img)

"""
create a numpy array with test images data and labels to evaluate and predict the results on test data
"""
def label_img(name):
    if name == 'FULL_VISIBILITY': return np.array([1, 0, 0])
    elif name == 'NO_VISIBILITY' : return np.array([0, 1, 0])
    elif name == 'PARTIAL_VISIBILITY': return np.array([0, 0, 1])

def load_test_data(path):
    test_data = []
    for clas in os.listdir(path):
        clas_path = os.path.join(path, clas)
        for file in os.listdir(clas_path):
            label = label_img(clas)

            file_path = os.path.join(clas_path, file)
            img = cv.imread(file_path, 0)
            img = img*(1./255) #apply normalization step that is applied in training
            ex_img = np.expand_dims(img, axis=-1)
            test_data.append([np.array(ex_img), label])

    shuffle(test_data)
    return test_data

"""
This class can be used to apply custom augmentation esp in case of imbalanced dataset
"""
class data_augmentation(object):
    def __init__(self):
        pass

    def merge_original_rotated_data(self, src_data_path, dst_data_path):
        files = os.listdir(src_data_path)
        for fname in files:
            src_file_path = os.path.join(src_data_path, fname)
            dst_file_path = os.path.join(dst_data_path, fname)

            shutil.copy(src_file_path, dst_file_path)

    def generate_aug_file_names(self, filename, ind):
        head, tail = os.path.split(filename)
        base, ext = tail.split('.')
        nfilename = base+'_'+str(ind)+'.'+ext
        file_path = os.path.join(head, nfilename)
        return file_path

    """
    @src_file_path: path fo the original image
    @aug_file_path: path where the augmented file needs to be copied
    @angle: roation angle
    @operation: what operation should be applied to augment the data
    """
    def augment_file(self, src_file_path, aug_file_path, angle, class_type, operation):
        img = cv.imread(src_file_path)
        b, g, r = cv.split(img)
        img_to_arr = tf.keras.preprocessing.image.img_to_array(img)
        sample = np.expand_dims(img_to_arr, 0)
        if operation == 'rotation':
            datagen = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=angle)
        elif operation == 'brightness':
            datagen = tf.keras.preprocessing.image.ImageDataGenerator(brightness_range=[0.2,1.0])
        elif operation == 'zoom':
            datagen = tf.keras.preprocessing.image.ImageDataGenerator(zoom_range=[0.5,1.0])

        no_of_samples_to_consider = 1
        if class_type == 'no_vis':
            no_of_samples_to_consider = 2

        it = datagen.flow(sample, batch_size = 1)
        if operation != 'rotation':
            for ind in range(no_of_samples_to_consider):
                batch_ = it.next()
                aug_file_path = self.generate_aug_file_names(aug_file_path, ind)
                cv.imwrite(aug_file_path, batch_[0].astype(np.uint8))
        else:
            batch_ = it.next()
            aug_file_path = self.generate_aug_file_names(aug_file_path, 0)
            cv.imwrite(aug_file_path, batch_[0].astype(np.uint8))

    """
    @aug_data_path: where augmented data needs to be stored
    @files_path: where all the original class data stored
    @angles: list contains all the angles that we want to rotate an image
    @operation: by default it is rotation
    """
    def trigger_augmentation(self, aug_data_path, files_path, angles, class_type, operation):
        """ Create a base path if does not exists already """
        if not os.path.exists(aug_data_path):
            os.makedirs(aug_data_path)

        """ Dump the images in particular operation for more clarity to visualize """
        aug_data_path = os.path.join(aug_data_path, operation)
        if not os.path.exists(aug_data_path):
            os.makedirs(aug_data_path)

        files = os.listdir(files_path)
        def process_aug(operation, angle=0):
            for fname in files:
                src_file_path = os.path.join(files_path, fname)

                """ create augment file path where it is needs to be saved"""
                base, ext = fname.split('.')
                if operation == 'rotation':
                    aug_file_name = base + '_rot_' + str(angle) + '.' + ext
                else:
                    aug_file_name = base + '_' + operation + '.' + ext
                aug_file_path = os.path.join(aug_data_path, aug_file_name)

                self.augment_file(src_file_path, aug_file_path, angle, class_type, operation)

        if operation == 'rotation':
            for angle in angles:
                process_aug(operation, angle)
        elif operation == 'zoom':
            process_aug(operation)
        elif operation == 'brightness':
            process_aug(operation)

"""
This class can be used to augment the data as train 60%, validation 20%, and test 20%
"""
class data_split(object):
    def __init__(self, orig_data_path):
        self.train_samples = 0
        self.valid_samples = 0
        self.test_samples = 0

        self.files = []
        self.src_data_path = orig_data_path
        self.class_src_data_path = ""

    """
    create the dest respective folders given high level path where classes path needs to be created
    @data_path: High level path where classes folders need to be created
    """
    def create_output_data_folders(self, data_path):
        if not os.path.exists(data_path):
            os.makedirs(data_path)

        fv_data_path = os.path.join(data_path, "FULL_VISIBILITY")
        if not os.path.exists(fv_data_path):
            os.makedirs(fv_data_path)

        pv_data_path = os.path.join(data_path, "PARTIAL_VISIBILITY")
        if not os.path.exists(pv_data_path):
            os.makedirs(pv_data_path)

        nv_data_path = os.path.join(data_path, "NO_VISIBILITY")
        if not os.path.exists(nv_data_path):
            os.makedirs(nv_data_path)

        return (fv_data_path, pv_data_path, nv_data_path)

    """
    copy the data in the given respective folders
    @test_samples: no of samples in test data folder
    @valid_samples: no of samples in validation data folder
    @train_samples: no of samples in training folder
    @files: total no of files in a class
    @test_data_path: path where test files needs to be copied
    @valid_data_path: path where validation files needs to be copied
    @train_data_path: path where training files needs to be copied
    """
    def copy_data_in_res_folders(self, test_data_path, valid_data_path, train_data_path):
        for _ in range(self.test_samples):
            rInd = np.random.randint(0, len(self.files)-1)
            src_file_name = self.files[rInd]
            src_file_path = os.path.join(self.class_src_data_path, src_file_name)

            head, tail = os.path.split(src_file_name)
            dst_file_name = os.path.join(test_data_path, tail)
            shutil.copy(src_file_path, dst_file_name)
            self.files.pop(rInd)

        for _ in range(self.valid_samples):
            rInd = np.random.randint(0, len(self.files)-1)
            src_file_name = self.files[rInd]
            src_file_path = os.path.join(self.class_src_data_path, src_file_name)

            head, tail = os.path.split(src_file_name)
            dst_file_name = os.path.join(valid_data_path, tail)
            shutil.copy(src_file_path, dst_file_name)
            self.files.pop(rInd)

        for ind in range(self.train_samples):
            src_file_name = self.files[ind]
            src_file_path = os.path.join(self.class_src_data_path, src_file_name)

            head, tail = os.path.split(src_file_name)
            dst_file_name = os.path.join(train_data_path, tail)
            shutil.copy(src_file_path, dst_file_name)

    """
    Given all the dataset, it create respectove class folder in training, validation, testing and copies files in respective folders
    @dst_path: high level path where folders and sub-folders needs to be created
    @class_type: specify class type to create folder associated to it
    """
    def split_util_func(self, dst_path, class_type):
        self.files = os.listdir(self.class_src_data_path)
        no_files = len(self.files)

        """ split data as 60%, 20% and 20%"""
        self.train_samples = int(no_files*0.6)
        self.valid_samples = int(no_files*0.2)
        self.test_samples = int(no_files*0.2)
        print(self.train_samples, self.valid_samples, self.test_samples, type(self.files))

        """ create destination sub folders"""
        """ train data path with diffferent classes"""
        train_data_path = os.path.join(dst_path, 'training')
        fv_train_data_path, pv_train_data_path, nv_train_data_path = self.create_output_data_folders(train_data_path)

        """ validation data path with diffferent classes"""
        validation_data_path = os.path.join(dst_path, 'validation')
        fv_valid_data_path, pv_valid_data_path, nv_valid_data_path = self.create_output_data_folders(validation_data_path)

        """ test data path with diffferent classes"""
        test_data_path = os.path.join(dst_path, 'test')
        fv_test_data_path, pv_test_data_path, nv_test_data_path = self.create_output_data_folders(test_data_path)

        if class_type == 'full_vis':
            self.copy_data_in_res_folders(fv_test_data_path, fv_valid_data_path, fv_train_data_path)
        elif class_type == 'partial_vis':
            self.copy_data_in_res_folders(pv_test_data_path, pv_valid_data_path, pv_train_data_path)
        elif class_type == 'no_vis':
            self.copy_data_in_res_folders(nv_test_data_path, nv_valid_data_path, nv_train_data_path)

    def split_data(self, sci_tvt_path):

        self.class_src_data_path = os.path.join(self.src_data_path, 'FULL_VISIBILITY')
        self.split_util_func(sci_tvt_path, 'full_vis')

        self.class_src_data_path = os.path.join(self.src_data_path, 'PARTIAL_VISIBILITY')
        self.split_util_func(sci_tvt_path, 'partial_vis')

        self.class_src_data_path = os.path.join(self.src_data_path, 'NO_VISIBILITY')
        self.split_util_func(sci_tvt_path, 'no_vis')


