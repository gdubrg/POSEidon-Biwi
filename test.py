from __future__ import division, print_function
import sys
import os

# arguments to chose if use cpu or gpu, and which gpu (number)
gpu_id = sys.argv[1]
cnmem = sys.argv[2]
print("Argument: gpu={}, mem={}".format(gpu_id, cnmem))
os.environ["THEANO_FLAGS"] = "device=" + gpu_id + ", lib.cnmem=" + cnmem

from loadBIWI import load_data, pre_load_val, load_data_OF_new as load_data_OF, load_data_gray

from model import FUSION
import numpy as np


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
b_standardization=True
b_data_augmentation=True
b_data_augmentation_val=True
f_angle_norm_method=-1
f_max_angle = 90.0
b_smooth = False
b_visual_weights = False
b_visual_input = False


nb_epoch = 1000
batch_size = 128
patience = 5
loss_check = 1000.0


# load model
model = FUSION(
    weights_depth   = "weights/weights_DEPTH.hdf5",
    weights_gray    = "weights/weights_GRAY.hdf5",
    weights_OF      = "weights/weights_OF.hdf5",
    depth_rows=rows, depth_cols=cols, gray_rows=rows, gray_cols=cols, of_rows=rows, of_cols=cols,
)


# load weight
print("Load weights ...")
model.load_weights('weights/weights_CONCAT.hdf5')

# load data
val_data = pre_load_val(path_data_of)
print(len(val_data))

X_OF, _ = load_data_OF(val_data, path_data_of, channels=of_channels, img_rows=rows, img_cols=cols,
                                selected_channels=of_sel_channels,
                                stretch_img=b_stretch,
                                standardize=b_standardization,
                                contrast=b_constrast_stretching,
                                equalize=b_equalize,
                                data_augmentation=False,
                                angle_norm_method=f_angle_norm_method,
                                max_angle=f_max_angle)

X_DEPTH, Y = load_data(val_data, path_data_depth, channels=channels, img_rows=rows, img_cols=cols,
                             stretch_img=b_stretch,
                             standardize=b_standardization,
                             contrast=b_constrast_stretching,
                             equalize=b_equalize,
                             data_augmentation=False,
                             angle_norm_method=f_angle_norm_method,
                             max_angle=f_max_angle)

X_GRAY, _ = load_data_gray(val_data, path_data_gray, channels=channels, img_rows=rows, img_cols=cols,
                                stretch_img=b_stretch,
                                standardize=b_standardization,
                                contrast=b_constrast_stretching,
                                equalize=b_equalize,
                                data_augmentation=False,
                                angle_norm_method=f_angle_norm_method,
                                max_angle=f_max_angle)

X_val = [X_DEPTH, X_GRAY, X_OF]


if b_visual_weights:
    import utility.get_filter_CNN as filt
    filt.get_filter(model, 0)
    filt.get_filter(model, 3)
    filt.get_filter(model, 6)


# prediction
pred = model.predict(x=X_val, batch_size=batch_size, verbose=1)
print(pred.shape)
print(np.min(pred))
print(np.max(pred))


# kalman filter
smooth_pred = np.zeros_like(pred)
if b_smooth:
    from pykalman import KalmanFilter

    for i in range(3):
        array_misure = pred[:, i]

        kf1D = KalmanFilter(
            transition_matrices=np.array([[1, 1], [0, 1]]),
            transition_covariance=1 * np.eye(2),
            observation_covariance=10000 * np.eye(1)
        )

        smoothed_pred = kf1D.smooth(array_misure)[0]
        smooth_pred[:, i] = smoothed_pred[:, 0]

# save predictions
f2 = open("results2_v3.txt", "w")
for i, p in enumerate(pred):
    line1 = val_data[i]['frame'] + "\t" + str(val_data[i]['angle1']) + "\t" + str(val_data[i]['angle2']) + "\t" + str(val_data[i]['angle3']) + "\n"

    if f_angle_norm_method != -1:
        line2 = val_data[i]['frame'] + "\t" + str((p[0] * (2*f_max_angle)) - f_max_angle) + "\t" + str((p[1] * (2*f_max_angle)) - f_max_angle) + "\t" + str((p[2] * (2*f_max_angle)) - f_max_angle) + "\n"
    else:
        line2 = val_data[i]['frame'] + "\t" + str(p[0] * (f_max_angle)) + "\t" + str(p[1] * (f_max_angle)) + "\t" + str(p[2] * (f_max_angle)) + "\n"

    f2.write(line1)
    f2.write(line2)
f2.close()

fsmooth = open("results_smooth.txt", "w")
for i, p in enumerate(smooth_pred):
    line1 = val_data[i]['frame'] + "\t" + str(val_data[i]['angle1']) + "\t" + str(val_data[i]['angle2']) + "\t" + str(val_data[i]['angle3']) + "\n"

    if f_angle_norm_method != -1:
        line2 = val_data[i]['frame'] + "\t" + str((p[0] * (2*f_max_angle)) - f_max_angle) + "\t" + str((p[1] * (2*f_max_angle)) - f_max_angle) + "\t" + str((p[2] * (2*f_max_angle)) - f_max_angle) + "\n"
    else:
        line2 = val_data[i]['frame'] + "\t" + str(p[0] * (f_max_angle)) + "\t" + str(p[1] * (f_max_angle)) + "\t" + str(p[2] * (f_max_angle)) + "\n"

    fsmooth.write(line1)
    fsmooth.write(line2)
fsmooth.close()

if b_visual_input:
    import utility.get_hidden_output as ho

    layer_output = ho.get_hidden_output(model, 0, X_val[40:50])
    ho.print_hidden_output(layer_output, 0)

    layer_output = ho.get_hidden_output(model, 1, X_val[40:50])
    ho.print_hidden_output(layer_output, 1)

    layer_output = ho.get_hidden_output(model, 2, X_val[40:50])
    ho.print_hidden_output(layer_output, 2)

    layer_output = ho.get_hidden_output(model, 3, X_val[40:50])
    ho.print_hidden_output(layer_output, 3)

    layer_output = ho.get_hidden_output(model, 4, X_val[40:50])
    ho.print_hidden_output(layer_output, 4)


