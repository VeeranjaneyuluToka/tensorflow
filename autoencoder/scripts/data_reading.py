import os
import numpy as np
import cv2 as cv`

"""
Movie data experimentation
"""
class data_reading(object):
    def __init__(self):
        pass

    """ Read image and resize it using opencv python """
    def get_img(self, path):
        img = cv.imread(path)
        resized_img = cv.resize(img, (128, 128))

        return resized_img

    """ Build train data """
    def load_train_data(self, path):
        x_train = []

        titles_allowed = 0
        for title_id in os.listdir(path):
            title_path = os.path.join(path, title_id)
            for fname in os.listdir(title_path):
                file_path = os.path.join(title_path, fname)
                x = self.get_img(file_path)
                x_train.append(x)

            titles_allowed += 1
            if titles_allowed > 5:
                break

        return x_train

    """ Build test data """
    def load_test_data(self, path):
        x_test = []

        titles_allowed = 0
        for title_id in os.listdir(path):
            title_path = os.path.join(path, title_id)
            for fname in os.listdir(title_path):
                file_path = os.path.join(title_path, fname)
                x = self.get_img(file_path)
                x_test.append(x)

            titles_allowed += 1
            if titles_allowed > 5:
                break

        return x_test

    """ load train and test data in the form of numpy arrays and normalize them, so that the range is in between (0, 1) """
    def create_data_numpy_arrays(self, training_data_path, testing_data_path):

        #load training data
        train_data = self.load_train_data(training_data_path)
        x_train_data = np.array(train_data)
        #print(x_train_data.shape)

        #load testing data
        test_data = self.load_test_data(testing_data_path)
        x_test_data = np.array(test_data)
        #print(x_test_data.shape)

        #reshape and normalize
        img_size = x_train_data.shape[1]
        x_train = np.reshape(x_train_data, [-1, img_size, img_size, 3])
        x_test = np.reshape(x_test_data, [-1, img_size, img_size, 3])
        x_train = x_train_data.astype('float32') / 255
        x_test = x_test_data.astype('float32') / 255
        #print(x_train.shape, x_test.shape)

        return img_size, x_train, x_test

    def create_test_data_randomly():
        src_path = "../data/movie_data/train/frames/"
        dst_path = "../data/movie_data/test_small/"

        file_names_list = []

        for fname in os.listdir(src_path):
                file_path = src_path+fname
                file_names_list.append(file_path)

        no_of_test_files = 200
        for i in range(no_of_test_files):
            ind = random.randint(0, len(file_names_list)-1)
            src_path = file_names_list[ind]
            file_names_list.pop(ind)
            fdir, fname = os.path.split(src_path)
            fdst_path = dst_path+fname
            shutil.move(src_path, fdst_path)

    def generate_data(directory, batch_size):
        i = 0
        file_list = os.listdir(directory)
        while True:
            img_batch = []
            for b in range(batch_size):
                if i == len(file_list):
                    i = 0
                    random.shuffle(file_list)
                sample = directory+file_list[i]
                i += 1
                #print("file path *****",sample)
                image = cv2.resize(cv2.imread(sample), (128, 128))
                if image is None:
                    print("image is none *******")
                img_batch.append((image.astype(float)-128)/128)
            yield np.array(img_batch)

    """
    A function to normalize the batch of data
    """
    def normalize(x):
        channel_axis = 3
        row_axis = 1
        col_axis = 2
        print(x.shape)
        featurewise_center = True
        featurewise_std_normalization=True
        if featurewise_center:
            mean = np.mean(x, axis=(0, row_axis, col_axis))
            broadcast_shape = [1, 1, 1]
            broadcast_shape[channel_axis - 1] = x.shape[channel_axis]
            mean = np.reshape(mean, broadcast_shape)
            x -= mean

    #split the data (given a dataset) as training and validation (80 and 20)
    def split_data_train_valid(path):
        no_of_files = len(os.listdir(path))
        valid_data_size = int(no_of_files * 0.2)
        train_data_size = no_of_files - valid_data_size
        #print(train_data_size, valid_data_size)

        file_names = []
        for fname in os.listdir(path):
            comp_file_path = path + fname
            file_names.append(comp_file_path)

        valid_data_path = '/mnt/disks/slow1/video_processing/exp/AE_arch/data/movie_data/bbs_frd_sv/valid/frames/'
        train_data_path = '/mnt/disks/slow1/video_processing/exp/AE_arch/data/movie_data/bbs_frd_sv/train/frames/'

        #move all the validation files to respective folder
        for i in range(valid_data_size):
            ind = random.randint(0, len(file_names)-1)
            src_path = file_names[ind]
            file_names.pop(ind)
        fdir, fname = os.path.split(src_path)
        fdst_path = valid_data_path+fname
        shutil.move(src_path, fdst_path)

        for i in range(train_data_size):
            src_path = file_names[i]
            fdir, fname = os.path.split(src_path)
            fdst_path = train_data_path+fname
            shutil.move(src_path, fdst_path)

    #count no.of files present in a given path
    def check_complete_data_size(path):
        count = 0
        for seg_fold in os.listdir(path):
            seg_path = path+seg_fold
            for seg_sub_fold in os.listdir(seg_path):
                seg_sub_path = seg_path+'/'+seg_sub_fold
                for fname in os.listdir(seg_sub_path):
                    if fname == 'imagecluster':
                        continue
                    count += 1
        print('total no.of images:',count)

    def generate_unique_data(path):
    global count
    no_of_files = len(os.listdir(path))
    file_names = []
    dst_path = '/mnt/disks/slow1/video_processing/frames/final_unique_data/'

    for fname in os.listdir(path):
        comp_file_path = path + '/' + fname
        file_names.append(comp_file_path)

    ind = 0
    if no_of_files <= 5: #consider only one frame
        src_file_name = file_names[ind]
        base, fname = os.path.split(src_file_name)
        f, ext = fname.split('.')
        dst_path_fname = dst_path + '/' + fname
        shutil.copy(src_file_name, dst_path_fname)
        new_file_path = dst_path + '/' + str(count)+'.'+ext
        os.rename(dst_path_fname, new_file_path)
        count += 1
    elif no_of_files > 5 and no_of_files <=20: #consider a frame for every 5 frame
        step = 5
        while ind < no_of_files:
            src_file_name = file_names[ind]
            base, fnmae = os.path.split(src_file_name)
            f, ext = fname.split('.')
            dst_path_fname = dst_path + '/' + fname
            shutil.copy(src_file_name, dst_path_fname)
            new_file_path = dst_path + '/' + str(count)+'.'+ext
            os.rename(dst_path_fname, new_file_path)
            count += 1
            ind += step
    elif no_of_files > 20: #consider 1 file for every 8 files
        step = 10
        while ind < no_of_files:
            src_file_name = file_names[ind]
            base, fname = os.path.split(src_file_name)
            f, ext = fname.split('.')
            dst_path_fname = dst_path+'/'+fname
            shutil.copy(src_file_name, dst_path_fname)
            new_file_path = dst_path + '/' + str(count)+'.'+ext
            os.rename(dst_path_fname, new_file_path)
            count += 1
            ind += step

    #copy last file
    src_file_name = file_names[no_of_files-1]
    base, fname = os.path.split(src_file_name)
    dst_path_fname = dst_path+'/'+fname
    shutil.copy(src_file_name, dst_path_fname)


    #parse all the clusters of data and create unique data set
def create_train_valid_from_clusters(path):
    for seg_fold in os.listdir(path):
        seg_path = path + seg_fold
        for sub_seg_fold in os.listdir(seg_path):
            sub_seg_path = seg_path + '/' + sub_seg_fold +'/'+'imagecluster'+'/'+'clusters'
            for clusters_fold in os.listdir(sub_seg_path):
                clusters_path = sub_seg_path + '/'+clusters_fold
                for cluster_fold in os.listdir(clusters_path):
                    final_cluster_path = clusters_path + '/' + cluster_fold
                    generate_unique_data(final_cluster_path)

#check if image is valid using opencv
def check_if_image_is_valid():
    for fname in os.listdir(path):
        comp_path = path+fname
        #print(comp_path)
        img = cv2.imread(comp_path)
        if img is None:
            print(comp_path)
            continue

        # tried spliting whole dataset as two halfs
def split_data_set(path):
    file_names = []

    for fname in os.listdir(path):
        com_fname = path + fname
        file_names.append(com_fname)

    print(len(file_names))
    sorted(file_names)
    new_path_1 = '/mnt/disks/slow1/video_processing/frames/breaking_bad_s01e07_1/'
    if not os.path.exists(new_path_1):
        os.makedirs(new_path_1)

    new_path_2 = '/mnt/disks/slow1/video_processing/frames/breaking_bad_s01e07_2/'
    if not os.path.exists(new_path_2):
        os.makedirs(new_path_2)

    end_ind = len(file_names)
    count = 0
    mid_ind = int(end_ind / 2)
    while count < end_ind:
        fname = file_names[count]
        head, tail = os.path.split(fname)

        if count < mid_ind:
            dst_path = new_path_1 + tail
        else:
            dst_path = new_path_2 + tail

        shutil.copy(fname, dst_path)
        count += 1

        #copy file from src to dst path and rename file which is in dst path
def copy_files(src_path, dst_path, rename_file_path):
    shutil.copy(src_path, dst_path)
    os.rename(dst_path, rename_file_path)

    def get_all_data_in_folder():
    path = '../../../frames/'
    dst_path = "../data/movie_data/train_bbs/frames/"
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)

    count = 0
    for sub_fold_name in os.listdir(path):
        comp_sub_fold_path = path+sub_fold_name
        print(comp_sub_fold_path)
        for fname in os.listdir(comp_sub_fold_path):
            src_file_path = comp_sub_fold_path+'/'+fname
            base, ext = fname.split('.')
            dst_file_path = dst_path+fname
            rename_file_path = dst_path + str(count) + ".jpg"
            copy_files(src_file_path, dst_file_path, rename_file_path)
            count += 1


            #create test data randomly picking up files
def create_test_data_randomly():
        file_names_list = []

        for fname in os.listdir(src_path):
                file_path = src_path+fname
                file_names_list.append(file_path)

        no_of_test_files = 5000
        for i in range(no_of_test_files):
                ind = random.randint(0, len(file_names_list)-1)
                src_path = file_names_list[ind]
                file_names_list.pop(ind)
                fdir, fname = os.path.split(src_path)
                fdst_path = dst_path+fname
                shutil.move(src_path, fdst_path)


                #to make predict_gen work, changed the path of original frames path
def create_new_folder_structure_predict_gen():
    parent_path = "/mnt/disks/slow1/video_processing/small_video_clip_frames_keras/small_video_clip_frames/"
    for sub_path in os.listdir(parent_path):
        sub_sub_path = parent_path + sub_path
        for fin_path in os.listdir(sub_sub_path):
            comp_path = sub_sub_path + '/'+ fin_path
            dst_path = comp_path + '/' + 'frames'
            if os.path.exists(dst_path):
                shutil.rmtree(dst_path)
            for f_name in os.listdir(comp_path):
                file_path = comp_path + '/' + f_name
                new_dst_path = comp_path + '/' + 'frames'
                if not os.path.exists(new_dst_path):
                    os.makedirs(new_dst_path)
                dst_file_path = new_dst_path + '/' + f_name
                shutil.move(file_path, dst_file_path)

                #suppose to remove older feature vectors folder
def remove_folders():
    parent_path = "/mnt/disks/slow1/video_processing/small_video_clip_frames_keras/small_video_clip_frames/"
    for sub_path in os.listdir(parent_path):
        comp_path_so_far = parent_path + sub_path
        for sub_sub_path in os.listdir(comp_path_so_far):
            final_sub_path = comp_path_so_far + '/' + sub_sub_path
            for fname in os.listdir(final_sub_path+'/'+'frames'):
                    if fname == 'feature_vectors_plf' or fname == 'feature_vectors' or fname == 'scalars':
                        fin_comp_path = final_sub_path+'/'+'frames'+'/'+fname
                        print(fin_comp_path)

                        #remove .json files which are in segmented folders to create clusters
def remove_unnecessary_files(path):
    for sub_fold in os.listdir(path):
        sub_fold_path = path+sub_fold
        for sub_sub_fold in os.listdir(sub_fold_path):
            sub_sub_path = sub_fold_path + '/' + sub_sub_fold
            for fname in os.listdir(sub_sub_path):
                comp_path = sub_sub_path + '/' + fname
                base, ext = fname.split('.')
                if ext != 'jpg':
                    print(comp_path)
                    os.remove(comp_path)
