***


# Classifying the visibility of ID cards in photos

The folder images inside data contains several different types of ID documents taken in different conditions and backgrounds. The goal is to use the images stored in this folder and to design an algorithm that identifies the visibility of the card on the photo (FULL_VISIBILITY, PARTIAL_VISIBILITY, NO_VISIBILITY).

## Data

Inside the data folder you can find the following:

### 1) Folder images
A folder containing the challenge images.

### 2) gicsd_labels.csv
A CSV file mapping each challenge image with its correct label.
	- **IMAGE_FILENAME**: The filename of each image.
	- **LABEL**: The label of each image, which can be one of these values: FULL_VISIBILITY, PARTIAL_VISIBILITY or NO_VISIBILITY. 


## Dependencies

setGPU - to select GPU core from available cores

python 3.7.3
tensorflow 1.14.0
OpenCV 4.1.1

seaborn
bm3d
matplotlib
numpy
pandas


## Run Instructions

python main.py  - triggers training with default arguments
python main.py --operatino='train' --csv_file_path='' --images_path='' - if you want to pass your own csv file and images path
for example: python main.py --operation='train' --csv_file_path='../data/gicsd_labels.csv' --images_path='../data/images'

python main.py --operation='predict' --file_path - displays the predictions with given file path
for example: python main.py --operation='predict' --file_path='../data/sci_tvt/test/NO_VISIBILITY/GICSD_8_7_213_SC.png'

## Approach

Data processing pipeline:
step 1 - have read data from csv and split them based on labels and visualize the distribution of each class, they are hugely imbalanced (plot shows that in notebook)
step 2. applied custom data augmentation to reduce the data imbalance (again plot shows the balanced dataset)
step 3. merged the augmeted data and original data, and extracted the sharp images from all the noisy dataset
step 4. Split the dataset as training, validation, and testing (60%, 20% and 20%)

There are some temporary folders gets created in this process

Architecture building and evaluation:
1. Have used tf.keras APIs to build architecture and to feed the data to the architecture
2. Have adjusted a few hyper parameters from the initial once, for example have used 32 as batch size and learning rate is 0.001, with these parameters network not converging and loss looks like Gaussian curve.
Then felt that 32 is too low as batch size as data might be a bit of duplicated in upsampling process, so used 64 and reduced learning rate to 0.0001 and used 'ReduceLROnPlateau' api to adjust LR
Initial built network is having 10 millians of parameters with which it is over-fitting, reduced no.of parameters to 847,763 (can notice the same in model summary) and used 0.5 dropout rate to avoid the overfitting the model

Training is stable and gives around 92% accuracy on validation set

predictions:
Have used the trianed model and  built the nd array from test data and evaluated on the testing data, it gave ~90 on test data as well.
showed confusion matrix with predictions in notebook. Predictions values are displayed as list of [FULL_VISIBILITY, NO_VISIBILITY, PARTIAL_VIBILITY
##### Future Work ####
Data processing:
Can use better augmentationt techniques such as
1. SMOTE
2. AutoAugment "AutoAugment:Learning Augmentation Strategies from Data", RandAugment "RandAugment: Practical automated data augmentation with a reduced search space"
These are proven that they gave a good results in recent state of the art classification techniques like EfficientNet architectures. They are good augmentation techniques becuase they find out appropriate filters by analysing data itself rather than applying manually.

We can explore with different data augmentation techniques and improve the augmentation step further which will really enhance the generalization of the model.

I still used ImageDataGenerator but using tf.data.Dataset APIs are beneficial when the dataset is large

Architecture:
Quickly built the architecture, however the following improvements can be added to improve further

1. k-fold cross validation to measure the models performance with different folds of data

2. Keras-tuner or HParams Dashboard to tune the hyper parameters further and improve the accuracy

3. Though imagenet dataset is biased towards the flowers, animals dataset, it might be useful to use the initial convbase rather then random initialization (that i have done here). Esp using latest pre-trained models like NoisyStudent version of EfficientNet and applying transfer learning should really improve the accuracy.

So there are quite a few options to improve further with latest techniques.