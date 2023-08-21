"""
Prediction of smoking variable. Setting weights inversely proportional to frequency.
"""
import pickle
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras import datasets, layers, models
from tensorflow.keras import backend as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dropout, Dense, Activation, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import sklearn
from keras_preprocessing.image import ImageDataGenerator
import tensorflow.keras as keras 
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.inception_v3 import InceptionV3
from sklearn.utils import resample
import scipy.stats as st
from tensorflow.keras.models import load_model
from itertools import chain
import argparse
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score



###INPUTS: PASTE IN SLURM###################
parser = argparse.ArgumentParser()
parser.add_argument("--num_epochs", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--n_gpus", type=int, default=1)
parser.add_argument("--cpu_cores", type=int, default=12)
parser.add_argument("--variable", type=str, default='Smoking')
parser.add_argument("--save_name", type=str, default='smoking')
parser.add_argument("--path_inputs", type=str, default='/scratch/fundus_data')
parser.add_argument("--path_outputs", type=str, default='/home/andreap/fundusdata_2/results_retina/')
args = parser.parse_args()
batch_size = args.batch_size
num_epochs = args.num_epochs
n_gpus = args.n_gpus
cpu_cores = args.cpu_cores
variable = args.variable
save_name = args.save_name
path_inputs = args.path_inputs
path_outputs = args.path_outputs


#Only Allocate GPU memory when needed - not upfront
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess =  tf.compat.v1.Session(config=config)


device_type = 'GPU'
devices = tf.config.experimental.list_physical_devices(
          device_type)
devices_names = [d.name.split("e:")[1] for d in devices]
strategy = tf.distribute.MirroredStrategy(
           devices=devices_names[:n_gpus])

strategy = tf.distribute.MirroredStrategy(
           devices=devices_names[:n_gpus])

device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
    raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))


#Load train, validation, test set image split
train_df=pd.read_csv(path_inputs + '/GoodQualityImages/Train/train_df.csv')
test_df=pd.read_csv(path_inputs + '/GoodQualityImages/Test/test_df.csv')
val_df=pd.read_csv(path_inputs + '/GoodQualityImages/Validation/val_df.csv')

#Convert to binary non smoking and smoking variable
train_df['Smoking_binarized'] = train_df['Smoking'].map({'current': 1, 'ideal': 0, 'intermediate' : 0})
val_df['Smoking_binarized'] = val_df['Smoking'].map({'current': 1, 'ideal': 0, 'intermediate' : 0})

#Data augmentation
datagen_training=ImageDataGenerator(
    rotation_range=20, 
    vertical_flip=True,
    horizontal_flip=True, 
    fill_mode="nearest", 
    preprocessing_function=tf.keras.applications.inception_v3.preprocess_input)
datagen_validation=ImageDataGenerator(
    preprocessing_function=tf.keras.applications.inception_v3.preprocess_input)

train_df_filtered = train_df[train_df['Smoking'].notna()]
val_df_filtered = val_df[val_df['Smoking'].notna()]
test_df_filtered = test_df[test_df['Smoking'].notna()]

#Setting weights inversely proportional to frequency
weights = {'non_smoking': 1, 'smoking': len(train_df_filtered[train_df_filtered['Smoking_binarized'] == 0])/len(train_df_filtered[train_df_filtered['Smoking_binarized'] == 1])}

train_df_filtered['Smoking_binarized_string'] = train_df_filtered['Smoking_binarized'].map({1: 'smoking', 0: 'non_smoking'})
val_df_filtered['Smoking_binarized_string'] = val_df_filtered['Smoking_binarized'].map({1: 'smoking', 0: 'non_smoking'})

train_generator=datagen_training.flow_from_dataframe(
    dataframe=train_df_filtered,
    directory= path_inputs + '/GoodQualityImages/Train/',
    x_col="Image_ID",
    y_col='Smoking_binarized_string',
    batch_size=batch_size,
    seed=42,
    shuffle=True,
    class_mode="binary",
    target_size=(587,587))

valid_generator=datagen_validation.flow_from_dataframe(
    dataframe=val_df_filtered,
    directory= path_inputs + '/GoodQualityImages/Validation/',
    x_col="Image_ID",
    y_col='Smoking_binarized_string',
    batch_size=batch_size,
    seed=42,
    shuffle=True,
    class_mode="binary",
    target_size=(587,587))

print('Class sorting train generator', train_generator.__dict__['class_indices'])
print('Class sorting val generator', valid_generator.__dict__['class_indices'])

weights = {0: weights[list(train_generator.__dict__['class_indices'].keys())[0]], 1: weights[list(train_generator.__dict__['class_indices'].keys())[1]]}

STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size


with strategy.scope():
    base_model = InceptionV3(input_shape = (587, 587, 3), include_top = False, weights = 'imagenet')

    for layer in base_model.layers:
        layer.trainable = True
    
    x = GlobalAveragePooling2D()(base_model.output)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    inception_classification = keras.Model(base_model.inputs, outputs)

    auc = tf.keras.metrics.AUC()

    inception_classification.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=tf.keras.optimizers.Adam(), metrics=[auc])
    
    filepath = path_outputs + save_name + '_singleinput.hdf5'
    callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath=filepath, 
        monitor='val_auc', verbose=0, save_best_only=True, mode='max'), 
        tf.keras.callbacks.EarlyStopping(monitor='val_auc', patience=50, verbose=0, mode='max')]

    inception_classification = inception_classification.fit(train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=num_epochs, callbacks = [callbacks], 
                    use_multiprocessing=False, workers = cpu_cores, 
                    max_queue_size=100, 
                    class_weight = weights)

#Load & Test Model
filepath = path_outputs + save_name + '_singleinput.hdf5'
smoking_oneiput_oneoutput = load_model(filepath)

test_datagen=ImageDataGenerator(preprocessing_function=tf.keras.applications.inception_v3.preprocess_input)

test_generator=test_datagen.flow_from_dataframe(
    dataframe=test_df_filtered,
    directory= path_inputs + '/GoodQualityImages/Test/',
    x_col="Image_ID",
    y_col=None,
    batch_size=1,
    seed=42,
    shuffle=False,
    class_mode=None,
    target_size=(587,587))

STEP_SIZE_TEST=test_generator.n//test_generator.batch_size

test_generator.reset()
y_predictions=smoking_oneiput_oneoutput.predict_generator(test_generator,
steps=STEP_SIZE_TEST,
verbose=1)

y_predictions_classes = [1 if x > 0.5 else 0 for x in y_predictions]
y_predictions_classes = np.array(y_predictions_classes)


test_df_filtered['Smoking_binarized'] = test_df_filtered['Smoking'].map({'current': 'smoking', 'ideal': 'non_smoking', 'intermediate' : 'non_smoking'})
test_df_filtered['Smoking_binarized'] = test_df_filtered['Smoking_binarized'].map({'smoking': train_generator.__dict__['class_indices']['smoking'], 'non_smoking': train_generator.__dict__['class_indices']['non_smoking']})
y_test = test_df_filtered['Smoking_binarized']


print('Accuracy:', accuracy_score(y_test, y_predictions_classes))
print('Precision Score:', precision_score(y_test, y_predictions_classes))
print('Recall Score:', recall_score(y_test, y_predictions_classes))
print('AUC:', roc_auc_score(y_test, y_predictions_classes))