import matplotlib as mpl
import keras.backend as K
from keras.layers import Dropout, Reshape, Flatten, LeakyReLU, Activation, Dense, BatchNormalization, SpatialDropout2D
from keras.layers.convolutional import Conv2D, SeparableConv2D, UpSampling2D, MaxPooling2D, AveragePooling2D
from keras.regularizers import L1L2
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from keras.datasets import cifar10
from libs.keras_adversarial.image_grid_callback import ImageGridCallback
from libs.keras_adversarial import AdversarialModel, simple_gan, gan_targets, fix_names
from libs.keras_adversarial import AdversarialOptimizerSimultaneous, normal_latent_sampling, AdversarialOptimizerAlternating
from libs.utils.image_utils import dim_ordering_unfix, dim_ordering_shape
import numpy as np
import pandas as pd
