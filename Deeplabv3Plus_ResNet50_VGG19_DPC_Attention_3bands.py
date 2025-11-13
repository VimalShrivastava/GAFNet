import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import cv2
from glob import glob
import matplotlib.pyplot as plt
import tifffile as tiff
from matplotlib import pyplot as plt
from PIL import Image
from tensorflow.keras.utils import to_categorical
#import tensorflow_datasets as tfds
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, BatchNormalization
from tensorflow.keras.layers import Activation, add, multiply, Lambda
from tensorflow.keras.layers import AveragePooling2D, average, UpSampling2D, Dropout
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.initializers import glorot_normal, random_normal, random_uniform
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from tensorflow.keras import backend as K
from tensorflow.keras.applications import VGG19, densenet
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
from tensorflow.keras.activations import sigmoid
from tensorflow.keras import layers
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import backend as keras
import random
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input, Dropout, Dense
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import models, layers, regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
from keras.metrics import MeanIoU
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense, multiply, Permute, Concatenate, Conv2D, Add, Activation, Lambda
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
from tensorflow.keras.activations import sigmoid
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tifffile as tiff
from tensorflow.keras.initializers import glorot_normal, he_normal
#kinit = 'glorot_normal'
kinit = 'he_normal'
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D
from tensorflow.keras.layers import Resizing
from tensorflow.keras.applications import Xception

def SqueezeAndExcite(inputs, ratio=8):
    """ Squeeze and Excitation Module """
    filters = inputs.shape[-1]
    se = GlobalAveragePooling2D()(inputs)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    return Multiply()([inputs, se])


def dense_prediction_cell(x, filters=256):
    """Dense Prediction Cell for multiscale feature enhancement"""
    # Branch 1
    b1 = layers.Conv2D(filters, 3, dilation_rate=1, padding='same', use_bias=False)(x)
    b1 = layers.BatchNormalization()(b1)
    b1 = layers.ReLU()(b1)

    # Branch 2
    b2 = layers.Conv2D(filters, 3, dilation_rate=3, padding='same', use_bias=False)(b1)
    b2 = layers.BatchNormalization()(b2)
    b2 = layers.ReLU()(b2)

    # Branch 3
    b3 = layers.Conv2D(filters, 3, dilation_rate=6, padding='same', use_bias=False)(b2)
    b3 = layers.BatchNormalization()(b3)
    b3 = layers.ReLU()(b3)

    # Branch 4 (uses b2 and b3)
    b4_input = layers.Concatenate()([b2, b3])
    b4 = layers.Conv2D(filters, 3, dilation_rate=9, padding='same', use_bias=False)(b4_input)
    b4 = layers.BatchNormalization()(b4)
    b4 = layers.ReLU()(b4)

    # Branch 5 (uses b1 and b4)
    b5_input = layers.Concatenate()([b1, b4])
    b5 = layers.Conv2D(filters, 3, dilation_rate=12, padding='same', use_bias=False)(b5_input)
    b5 = layers.BatchNormalization()(b5)
    b5 = layers.ReLU()(b5)

    # Final aggregation
    concat = layers.Concatenate()([b1, b2, b3, b4, b5])
    out = layers.Conv2D(filters, 1, padding='same', use_bias=False)(concat)
    out = layers.BatchNormalization()(out)
    out = layers.ReLU()(out)

    return out


def atrous_spatial_pyramid_pooling(x):
    """ ASPP Module for multi-scale context aggregation """
    d1 = Conv2D(256, (1, 1), dilation_rate=1, padding="same", activation="relu")(x)
    d2 = Conv2D(256, (3, 3), dilation_rate=6, padding="same", activation="relu")(x)
    d3 = Conv2D(256, (3, 3), dilation_rate=12, padding="same", activation="relu")(x)
    d4 = Conv2D(256, (3, 3), dilation_rate=18, padding="same", activation="relu")(x)

    aspp_out = Concatenate()([d1, d2, d3, d4])
    return Conv2D(256, (1, 1), padding="same", activation="relu")(aspp_out)

def attention_block(x, gating, inter_shape):
    shape_x = K.int_shape(x)
    shape_g = K.int_shape(gating)

# Getting the x signal to the same shape as the gating signal
    theta_x = layers.Conv2D(inter_shape, (2, 2), strides=(2, 2), padding='same', kernel_initializer=kinit)(x)  # 16
    shape_theta_x = K.int_shape(theta_x)

# Getting the gating signal to the same number of filters as the inter_shape
    phi_g = layers.Conv2D(inter_shape, (1, 1), padding='same', kernel_initializer=kinit)(gating)
    upsample_g = layers.Conv2DTranspose(inter_shape, (3, 3),
                                 strides=(shape_theta_x[1] // shape_g[1], shape_theta_x[2] // shape_g[2]),
                                 padding='same', kernel_initializer=kinit)(phi_g)  # 16

    concat_xg = layers.add([upsample_g, theta_x])
    act_xg = layers.Activation('relu')(concat_xg)
    psi = layers.Conv2D(1, (1, 1), padding='same', kernel_initializer=kinit)(act_xg)
    sigmoid_xg = layers.Activation('sigmoid')(psi)
    shape_sigmoid = K.int_shape(sigmoid_xg)
    upsample_psi = layers.UpSampling2D(size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2]))(sigmoid_xg)  # 32

    upsample_psi = repeat_elem(upsample_psi, shape_x[3])

    y = layers.multiply([upsample_psi, x])

    result = layers.Conv2D(shape_x[3], (1, 1), padding='same', kernel_initializer=kinit)(y)
    result_bn = layers.BatchNormalization()(result)
    return result_bn

def repeat_elem(tensor, rep):
    # lambda function to repeat Repeats the elements of a tensor along an axis
    #by a factor of rep.
    # If tensor has shape (None, 256,256,3), lambda will return a tensor of shape
    #(None, 256,256,6), if specified axis=3 and rep=2.

     return layers.Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3),
                          arguments={'repnum': rep})(tensor)


def deeplabv3plus_resnet50_vgg19_dpc_attention_3band(input_shape1, num_classes):
    """ Input Layers """
    input_1 = Input(input_shape1)  # RGB Image (512,512,3)
    #input_2 = Input(input_shape2)  # Grayscale Image (512,512,1)

    """ Encoder (ResNet50 for RGB) """
    encoder = ResNet50(weights="imagenet", include_top=False, input_tensor=input_1)
    encoder_vgg = VGG19(weights="imagenet", include_top=False, input_tensor=input_1)

    x_a_org = encoder.get_layer("conv4_block6_out").output   # (None, 32, 32, 1024)
    x_a = dense_prediction_cell(x_a_org, 256)  # Apply DPC
    x_a = UpSampling2D((4, 4), interpolation="bilinear")(x_a)  # (None, 128, 128, 256)

    x_a_vgg_org = encoder_vgg.get_layer("block5_conv4").output   # (None, 32, 32, 1024)
    #x_a_vgg = atrous_spatial_pyramid_pooling(x_a_vgg_org)  # Apply ASPP
    x_a_vgg = dense_prediction_cell(x_a_vgg_org, 256)  # Apply DPC
    x_a_vgg = UpSampling2D((4, 4), interpolation="bilinear")(x_a_vgg) # (None, 128, 128, 1024)


    #print(x_a.shape)
    mid_features_org = encoder.get_layer("conv3_block4_out").output  # Mid-level features (None, 64, 64, 512)
    mid_features = dense_prediction_cell(mid_features_org, 256)  # ASPP
    mid_features = UpSampling2D((2, 2), interpolation="bilinear")(mid_features)  # Mid-level features (None, 128, 128, 512)

    mid_features_vgg_org = encoder_vgg.get_layer("block4_conv4").output  # Mid-level features (None, 64, 64, 512)
    #mid_features_vgg = atrous_spatial_pyramid_pooling(mid_features_vgg_org)
    mid_features_vgg = dense_prediction_cell(mid_features_vgg_org, 256)
    mid_features_vgg = UpSampling2D((2, 2), interpolation="bilinear")(mid_features_vgg)  # Mid-level features (None, 128, 128, 512)

    x_b_org = encoder.get_layer("conv2_block3_out").output  # Shallow features (None, 128, 128, 256)
    x_b = dense_prediction_cell(x_b_org, 256)

    x_b_vgg_org = encoder_vgg.get_layer("block3_conv3").output  # Shallow features (None, 128, 128, 256)
    #x_b_vgg = atrous_spatial_pyramid_pooling(x_b_vgg_org)
    x_b_vgg = dense_prediction_cell(x_b_vgg_org, 256)

    attention_features_1 = attention_block( mid_features_org, x_a_org, 256)
    attention_features_1 = UpSampling2D ((2, 2), interpolation="bilinear")(attention_features_1)
    attention_features_2 = attention_block(x_b_org, mid_features_org, 256)

    attention_features_1_vgg = attention_block( mid_features_vgg_org, x_a_vgg_org, 256)
    attention_features_1_vgg = UpSampling2D ((2, 2), interpolation="bilinear")(attention_features_1_vgg)
    attention_features_2_vgg = attention_block( x_b_vgg_org, mid_features_vgg_org, 256)


    """ Concatenate RGB and Grayscale Features """
    x1 = Concatenate()([x_a,  mid_features, x_b, attention_features_1, attention_features_2])  # Shapes now match (None, 128, 128, 1024)
    x1 = Conv2D(filters=128, kernel_size=3, padding='same', use_bias=False)(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x = SqueezeAndExcite(x1)


    x1_vgg = Concatenate()([x_a_vgg, mid_features_vgg, x_b_vgg, attention_features_1_vgg, attention_features_2_vgg])  # Shapes now match (None, 128, 128, 1024)
    x1_vgg = Conv2D(filters=256, kernel_size=3, padding='same', use_bias=False)(x1_vgg)
    x1_vgg = BatchNormalization()(x1_vgg)
    x1_vgg = Activation('relu')(x1_vgg)
    x_vgg = SqueezeAndExcite(x1_vgg)

    x1 = Concatenate()([x, x1, x_vgg, x1_vgg])

    x1 = Conv2D(filters=256, kernel_size=3, dilation_rate=2, padding='same', use_bias=False)(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)


    x = Conv2D(filters=512, kernel_size=3, dilation_rate=4, padding='same', use_bias=False)(x1)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)


    x = UpSampling2D((4, 4), interpolation="bilinear")(x)  # Restore full size (512,512)
    x = Conv2D(num_classes, kernel_size=(1, 1), padding="same")(x)
    x = Activation('softmax')(x)

    #model = Model(input_1, x)
    model = Model(inputs=input_1, outputs= x )
    return model

if __name__ == "__main__":
    input_shape1 = (512,512,3)  # RGB Image
    num_classes = 5            #LandCover Classes
    model = deeplabv3plus_resnet50_vgg19_dpc_attention_3band(input_shape1, num_classes)
    model.summary()