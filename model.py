from __future__ import division, print_function
from keras.models import Model
from keras.layers import Flatten, Dense, Dropout, Activation, Input, merge
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
import h5py


def get_weights(f, i, layer_name=None):
    if 'layer_names' not in f.attrs and 'model_weights' in f:
        f = f['model_weights']
    if layer_name is None:
        g = f['convolution2d_{}'.format(i)]
    else:
        g = f[layer_name + '_{}'.format(i)]
    weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]
    weight_values = [g[weight_name] for weight_name in weight_names]
    return weight_values


def CNN_DEPTH(input_img, filepath, out_layer=7):

    f = h5py.File(filepath, mode='r')

    weights = get_weights(f, 1)
    x = Convolution2D(30, 5, 5, trainable=False, weights=weights, subsample=(1, 1), activation='tanh')(input_img)
    conv1 = MaxPooling2D((2, 2))(x)

    weights = get_weights(f, 2)
    x = Convolution2D(30, 5, 5, trainable=False, weights=weights, subsample=(1, 1), activation='tanh')(conv1)
    conv2 = MaxPooling2D((2, 2))(x)

    weights = get_weights(f, 3)
    x = Convolution2D(30, 4, 4, trainable=False, weights=weights, subsample=(1, 1), activation='tanh')(conv2)
    conv3 = MaxPooling2D((2, 2))(x)

    weights = get_weights(f, 4)
    conv4 = Convolution2D(30, 3, 3, trainable=False, weights=weights, subsample=(1, 1), activation='tanh')(conv3)

    weights = get_weights(f, 5)
    conv5 = Convolution2D(120, 3, 3, trainable=False, weights=weights, subsample=(1, 1), activation='tanh')(conv4)

    x = Flatten()(conv5)

    weights = get_weights(f, 1, "dense")
    dense1 = Dense(120, trainable=False, weights=weights, activation='tanh')(x)
    # x = Dropout(0.5)(x)

    weights = get_weights(f, 2, "dense")
    dense2 = Dense(84, trainable=False, weights=weights, activation='tanh')(dense1)
    # x = Dropout(0.5)(x)

    weights = get_weights(f, 3, "dense")
    out = Dense(3, activation='tanh')(dense2)

    if out_layer not in range(1,8):
        raise Exception("Output layer is out of range 1-7")

    if out_layer == 1:
        return conv1
    elif out_layer == 2:
        return conv2
    elif out_layer == 3:
        return conv3
    elif out_layer == 4:
        return conv4
    elif out_layer == 5:
        return conv5
    elif out_layer == 6:
        return dense1
    elif out_layer == 7:
        return dense2

    return x


def FUSION_DENSE_CONCAT(weights_depth, weights_gray, weights_OF, depth_rows, depth_cols, gray_rows, gray_cols, of_rows, of_cols):

    depth_channel = 1
    ffd_channel = 1
    of_channel = 2

    input_depth = Input(shape=(depth_channel, depth_rows, depth_cols))
    input_gray = Input(shape=(ffd_channel, gray_rows, gray_cols))
    input_OF = Input(shape=(of_channel, of_rows, of_cols))

    cnn_depth_out = CNN_DEPTH(input_depth, weights_depth)
    cnn_of_out = CNN_DEPTH(input_OF, weights_OF)
    cnn_gray_out = CNN_DEPTH(input_gray, weights_gray)

    x = merge([cnn_depth_out, cnn_gray_out, cnn_of_out], mode='concat', concat_axis=-1, name="merge concat")

    x = Dense(120, activation='tanh', init='normal')(x)
    x = Dropout(0.5)(x)
    x = Dense(84, activation='tanh', init='normal')(x)
    x = Dropout(0.5)(x)
    out = Dense(3, activation='tanh', init='normal')(x)

    model = Model(input=[input_depth, input_gray, input_OF], output=out)

    return model


def FUSION(weights_depth, weights_gray, weights_OF, depth_rows, depth_cols, gray_rows, gray_cols, of_rows, of_cols):

    model = FUSION_DENSE_CONCAT(weights_depth, weights_gray, weights_OF, depth_rows, depth_cols, gray_rows, gray_cols, of_rows, of_cols)

    return model
