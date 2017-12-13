from __future__ import division, print_function
import sys
import os

# arguments to chose if use cpu or gpu, and which gpu (number)
gpu_id = sys.argv[1]
cnmem = sys.argv[2]
print("Argument: gpu={}, mem={}".format(gpu_id, cnmem))
os.environ["THEANO_FLAGS"] = "device=" + gpu_id + ", lib.cnmem=" + cnmem

from model import FUSION
import random
import numpy as np
from matplotlib import pyplot as plt
import logging
import datetime
from loadBIWI import pre_load, pre_load_val, load_data, load_data_val, load_data_OF_new as load_data_OF, load_data_val_OF_new as load_data_val_OF, load_data_gray, load_data_val_gray
from custom_loss import euclidean_distance_angles_biwi
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras as k


now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")

# set seed for shuffle
random.seed(1769)

# log file
logging.basicConfig(filename='log/training_output' + now + '.txt', level=logging.INFO)

def to_log(message):
    logging.info(message)

# pplot the loss during training phase
class LossHistory(k.callbacks.Callback):

    def __init__(self):
        plt.ion()
        fig = plt.figure()
        self.plot_loss = fig.add_subplot(211)
        self.plot_val_loss = fig.add_subplot(212)

    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []
        self.val_loss = 0
        self.loss = 0

    def on_epoch_end(self, epoch, logs={}):

        self.val_loss = logs.get('val_loss')
        self.loss = logs.get('loss')
        self.losses.append(self.loss)
        self.val_losses.append(self.val_loss)

        to_log("Training Loss: "+str(self.loss))
        to_log("Validation Loss: "+str(self.val_loss))

        self.plot_loss.plot(self.val_losses, 'r')
        self.plot_val_loss.plot(self.losses, 'r')

        plt.draw()
        plt.pause(0.0001)



# #####################################################################################################################
# MODEL PARAMETERS
# #####################################################################################################################

# comments
version_comment = 'FUSION-DEPTH-GRAY-OF-BIWI'
print("Comment: ", version_comment)

# dataset paths
path_data_depth = "Dataset/BIWI/face_dataset_large/"
path_data_gray = "Dataset/BIWI_GRAY/face_dataset_ae/"
path_data_of = "Dataset/BIWI/face_dataset_OF/"
print("Dataset DEPTH:", path_data_depth)
print("Dataset GRAY:", path_data_gray)
print("Dataset OF:", path_data_of)

# input
of_channels = 2
channels = 1
rows = 64
cols = 64
of_sel_channels = [0, 1]

# preprocessing options
b_stretch = False
b_equalize = False
b_constrast_stretching = True
b_standardization = True
b_data_augmentation = True
b_data_augmentation_val = True
f_angle_norm_method = -1
f_max_angle = 90.0

# train parameters
nb_epoch = 1000
batch_size = 128
patience = 15
loss_check = 1000.0

# import model
model = FUSION(
    weights_depth   = "weights/weights_DEPTH.hdf5",
    weights_gray    = "weights/weights_GRAY.hdf5",
    weights_OF      = "weights/weights_OF.hdf5",
    depth_rows=rows, depth_cols=cols, gray_rows=rows, gray_cols=cols, of_rows=rows, of_cols=cols,
)

# optimizer
sgd = SGD(lr=2e-2, decay=0.0005, momentum=0.9, nesterov=True)

# compile model
to_log("Compile model ...")
model.compile(optimizer=sgd, loss=euclidean_distance_angles_biwi)



# #####################################################################################################################
# Load data
# #####################################################################################################################

train_data = pre_load(path_data_of)
val_data = pre_load_val(path_data_of)

# data augmentation in train
if b_data_augmentation:
    n_repeat = 11
    train_data = [x.copy() for item in train_data for x in np.repeat(item, n_repeat)]
    [x.update({'data_augmentation': i % n_repeat}) for i,x in enumerate(train_data)]

# data augmentation in validation
if b_data_augmentation_val:
    val_data = [x for item in val_data for x in np.repeat(item, 6)]


# load validation data (Optical Flow)
X_val_OF, _ = load_data_val_OF(val_data, path_data_of, channels=of_channels, img_rows=rows, img_cols=cols,
                                selected_channels=of_sel_channels,
                                stretch_img=b_stretch,
                                standardize=b_standardization,
                                contrast=b_constrast_stretching,
                                equalize=b_equalize,
                                data_augmentation=b_data_augmentation_val,
                                angle_norm_method=f_angle_norm_method,
                                max_angle=f_max_angle)

# load validation data (depth)
X_val_DEPTH, Y_val = load_data_val(val_data, path_data_depth, channels=channels, img_rows=rows, img_cols=cols,
                             stretch_img=b_stretch,
                             standardize=b_standardization,
                             contrast=b_constrast_stretching,
                             equalize=b_equalize,
                             data_augmentation=b_data_augmentation_val,
                             angle_norm_method=f_angle_norm_method,
                             max_angle=f_max_angle)

# load validation data (Face-from-Depth data)
X_val_GRAY, _ = load_data_val_gray(val_data, path_data_gray, channels=channels, img_rows=rows, img_cols=cols,
                                stretch_img=b_stretch,
                                standardize=b_standardization,
                                contrast=b_constrast_stretching,
                                equalize=b_equalize,
                                data_augmentation=b_data_augmentation_val,
                                angle_norm_method=f_angle_norm_method,
                                max_angle=f_max_angle)

X_val = [X_val_DEPTH, X_val_GRAY, X_val_OF]
to_log("Training Data: %d" % len(train_data))
to_log("Validation Data: %d" % len(val_data))


# #####################################################################################################################
# Logging
# #####################################################################################################################

file_validation = open("log/info_network" + now + ".txt", "w")
try:
    file_validation.write("Argument:\t gpu={}, mem={}\n".format(gpu_id, cnmem))
    file_validation.write("Version:\t {}\n".format(version_comment))

    file_validation.write("Dataset DEPTH:\t {}\n".format(path_data_depth))
    file_validation.write("Dataset GRAY:\t {}\n".format(path_data_gray))
    file_validation.write("Dataset OF:\t {}\n".format(path_data_of))
    file_validation.write("Input:\t channel={}, rows={}, cols={}\n".format(channels, rows, cols))
    file_validation.write("Training:\t epoch={}, batch_size={}\n".format(nb_epoch, batch_size))

    file_validation.write("Training Data:\t %d \n" % len(train_data))
    file_validation.write("Validation Data:\t %d \n" % len(val_data))

    file_validation.write("Optimizer:\t " + str(model.optimizer) + "\n")
    file_validation.write("Learning Rate:\t {}\n".format(model.optimizer.lr.get_value()))
    file_validation.write("Decay:\t {}\n".format(model.optimizer.decay.get_value()))
    file_validation.write("Momentum:\t {}\n".format(model.optimizer.momentum.get_value()))
    file_validation.write("Nestorov:\t {}\n".format(model.optimizer.nesterov))
    file_validation.write("Loss:\t " + str(model.loss) + "\n")
except Exception as e:
    print("Error during file writing: {}".format(e.message))
finally:
    file_validation.flush()
    file_validation.close()
file_validation = open("log/info_network" + now + ".txt", "a")

file_validation.close()


# #####################################################################################################################
# TRAINING
# #####################################################################################################################

def generator():

    # shuffle data
    random.shuffle(train_data)

    while True:
        # load a batch of the training data (Optical Flow, depth and Fce-from-Depth)
        for it in range(0, len(train_data), batch_size):
            X_OF, _ = load_data_OF(train_data[it:it + batch_size], path_data_of, channels=of_channels, img_rows=rows, img_cols=cols,
                                selected_channels=of_sel_channels,
                                 stretch_img=b_stretch,
                                 standardize=b_standardization,
                                 contrast=b_constrast_stretching,
                                 equalize=b_equalize,
                                 data_augmentation=b_data_augmentation,
                                 angle_norm_method=f_angle_norm_method,
                                 max_angle=f_max_angle)
            X_DEPTH, Y = load_data(train_data[it:it + batch_size], path_data_depth, channels=channels, img_rows=rows, img_cols=cols,
                             stretch_img=b_stretch,
                             standardize=b_standardization,
                             contrast=b_constrast_stretching,
                             equalize=b_equalize,
                             data_augmentation=b_data_augmentation,
                             angle_norm_method=f_angle_norm_method,
                             max_angle=f_max_angle)
            X_GRAY, _ = load_data_gray(train_data[it:it + batch_size], path_data_gray, channels=channels, img_rows=rows,
                                  img_cols=cols,
                                  stretch_img=b_stretch,
                                  standardize=b_standardization,
                                  contrast=b_constrast_stretching,
                                  equalize=b_equalize,
                                  data_augmentation=b_data_augmentation,
                                  angle_norm_method=f_angle_norm_method,
                                  max_angle=f_max_angle)

            yield [X_DEPTH, X_GRAY, X_OF], Y


his = LossHistory()
model.fit_generator(generator(),
                    len(train_data),
                    nb_epoch,
                    validation_data=[X_val, Y_val],
                    callbacks=[his, EarlyStopping(patience=patience), ModelCheckpoint("weights/weights.{epoch:03d}-{val_loss:.5f}.hdf5", save_best_only=True)])
