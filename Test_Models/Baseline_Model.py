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
from functions import test_regression_model


#Load dataframe
test_df=pd.read_csv('/home/andreap/fundusdata_2/GoodQualityImages/Test/test_df.csv')


#Sort test df
test_df_filtered = test_df.sort_values(['eid', 'Image_ID'])


left_right = [x[8:13] for x in list(test_df_filtered['Image_ID'])]
test_df_filtered['left_right'] = left_right
test_df_filtered['Genetic sex_0.0_integers'] = test_df_filtered['Genetic sex_0.0'].map({'Female': 0, 'Male': 1})

test_df_filtered['Smoking_binarized'] = test_df_filtered['Smoking'].map({'current': 1, 'ideal': 0, 'intermediate' : 0})
#test_df_filtered['Smoking_binarized_string'] = test_df_filtered['Smoking_binarized'].map({1: 'smoking', 0: 'non_smoking'})

#Majority class predictions -> Female
test_df_filtered['naive_predictions_gender'] = 0
test_df_filtered['naive_predictions_smoking'] = 0

#DF Left Eye Images

test_df_filtered_gender = test_df_filtered[test_df_filtered['Genetic sex_0.0'].notna()]

df_left_gender = test_df_filtered_gender[test_df_filtered_gender['left_right'] == '21015']

#DF Right Eye Images
df_right_gender = test_df_filtered_gender[test_df_filtered_gender['left_right'] == '21016']

#DF Mean Eye Images
mean_df_gender = test_df_filtered_gender.groupby('eid').mean()




print('###Gender###')
print('Accuracy All Images:', accuracy_score(test_df_filtered_gender['Genetic sex_0.0_integers'], test_df_filtered_gender['naive_predictions_gender']))
print('Precision Score All Images:', precision_score(test_df_filtered_gender['Genetic sex_0.0_integers'], test_df_filtered_gender['naive_predictions_gender']))
print('Recall Score All Images:', recall_score(test_df_filtered_gender['Genetic sex_0.0_integers'], test_df_filtered_gender['naive_predictions_gender']))
print('AUC All Images:', roc_auc_score(test_df_filtered_gender['Genetic sex_0.0_integers'], test_df_filtered_gender['naive_predictions_gender']))

print('Accuracy Right Images:', accuracy_score(df_right_gender['Genetic sex_0.0_integers'], df_right_gender['naive_predictions_gender']))
print('Precision Score Right Images:', precision_score(df_right_gender['Genetic sex_0.0_integers'], df_right_gender['naive_predictions_gender']))
print('Recall Score Right Images:', recall_score(df_right_gender['Genetic sex_0.0_integers'], df_right_gender['naive_predictions_gender']))
print('AUC Right Images:', roc_auc_score(df_right_gender['Genetic sex_0.0_integers'], df_right_gender['naive_predictions_gender']))

print('Accuracy Left Images:', accuracy_score(df_left_gender['Genetic sex_0.0_integers'], df_left_gender['naive_predictions_gender']))
print('Precision Score Left Images:', precision_score(df_left_gender['Genetic sex_0.0_integers'], df_left_gender['naive_predictions_gender']))
print('Recall Score Left Images:', recall_score(df_left_gender['Genetic sex_0.0_integers'], df_left_gender['naive_predictions_gender']))
print('AUC Left Images:', roc_auc_score(df_left_gender['Genetic sex_0.0_integers'], df_left_gender['naive_predictions_gender']))

print('Accuracy Mean Images:', accuracy_score(mean_df_gender['Genetic sex_0.0_integers'], mean_df_gender['naive_predictions_gender']))
print('Precision Score Mean Images:', precision_score(mean_df_gender['Genetic sex_0.0_integers'], mean_df_gender['naive_predictions_gender']))
print('Recall Score Mean Images:', recall_score(mean_df_gender['Genetic sex_0.0_integers'], mean_df_gender['naive_predictions_gender']))
print('AUC Mean Images:', roc_auc_score(mean_df_gender['Genetic sex_0.0_integers'], mean_df_gender['naive_predictions_gender']))

test_df_filtered_smoking = test_df_filtered[test_df_filtered['Smoking'].notna()]

df_left_smoking = test_df_filtered_smoking[test_df_filtered_smoking['left_right'] == '21015']

#DF Right Eye Images
df_right_smoking = test_df_filtered_smoking[test_df_filtered_smoking['left_right'] == '21016']

#DF Mean Eye Images
mean_df_smoking = test_df_filtered_smoking.groupby('eid').mean()


print('###Smoking###')
print('Accuracy All Images:', accuracy_score(test_df_filtered_smoking['Smoking_binarized'], test_df_filtered_smoking['naive_predictions_smoking']))
print('Precision Score All Images:', precision_score(test_df_filtered_smoking['Smoking_binarized'], test_df_filtered_smoking['naive_predictions_smoking']))
print('Recall Score All Images:', recall_score(test_df_filtered_smoking['Smoking_binarized'], test_df_filtered_smoking['naive_predictions_smoking']))
print('AUC All Images:', roc_auc_score(test_df_filtered_smoking['Smoking_binarized'], test_df_filtered_smoking['naive_predictions_smoking']))

print('Accuracy Right Images:', accuracy_score(df_right_smoking['Smoking_binarized'], df_right_smoking['naive_predictions_smoking']))
print('Precision Score Right Images:', precision_score(df_right_smoking['Smoking_binarized'], df_right_smoking['naive_predictions_smoking']))
print('Recall Score Right Images:', recall_score(df_right_smoking['Smoking_binarized'], df_right_smoking['naive_predictions_smoking']))
print('AUC Right Images:', roc_auc_score(df_right_smoking['Smoking_binarized'], df_right_smoking['naive_predictions_smoking']))

print('Accuracy Left Images:', accuracy_score(df_left_smoking['Smoking_binarized'], df_left_smoking['naive_predictions_smoking']))
print('Precision Score Left Images:', precision_score(df_left_smoking['Smoking_binarized'], df_left_smoking['naive_predictions_smoking']))
print('Recall Score Left Images:', recall_score(df_left_smoking['Smoking_binarized'], df_left_smoking['naive_predictions_smoking']))
print('AUC Left Images:', roc_auc_score(df_left_smoking['Smoking_binarized'], df_left_smoking['naive_predictions_smoking']))

print('Accuracy Left Images:', accuracy_score(mean_df_smoking['Smoking_binarized'], mean_df_smoking['naive_predictions_smoking']))
print('Precision Score Left Images:', precision_score(mean_df_smoking['Smoking_binarized'], mean_df_smoking['naive_predictions_smoking']))
print('Recall Score Left Images:', recall_score(mean_df_smoking['Smoking_binarized'], mean_df_smoking['naive_predictions_smoking']))
print('AUC Left Images:', roc_auc_score(mean_df_smoking['Smoking_binarized'], mean_df_smoking['naive_predictions_smoking']))

print('###SBP###')
test_regression_model('Systolic blood pressure mean_0.0', test_df_filtered)

print('###DBP###')
test_regression_model('Diastolic blood pressure mean_0.0', test_df_filtered)

print('###BMI###')
test_regression_model('Body mass index (BMI)_0.0', test_df_filtered)

print('###Age###')
test_regression_model('Age', test_df_filtered)

print('###Cholesterol###')
test_regression_model('Cholesterol_0.0', test_df_filtered)

print('###HbA1c###')
test_regression_model('Glycated haemoglobin (HbA1c)_0.0', test_df_filtered)