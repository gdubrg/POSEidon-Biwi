from keras.backend import tensorflow_backend as K
from theano import tensor as T
import theano
import numpy as np

# weighted euclidean loss
def euclidean_distance_angles_biwi(y_true, y_pred):

    diff = y_pred - y_true
    weights = theano.shared(np.expand_dims(3 * np.array([0.2, 0.35, 0.45]), axis=0))
    weights = T.patternbroadcast(weights, (True, False))
    diff = diff * weights

    return K.sqrt(K.sum(K.square(diff), axis=-1, keepdims=True))

