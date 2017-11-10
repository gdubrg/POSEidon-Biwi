from __future__ import division, print_function
import numpy as np
import cv2
from rowStretching import stretch
from sklearn import preprocessing
from skimage import exposure, util
import random

random.seed(1834)

def pre_load(path, tst_id=-1, filter_angles=False):
    data = []

    if not isinstance(tst_id, list):
        if tst_id < 0:
            ids = range(1, 25)
            ids.remove(11)
            ids.remove(12)
        else:
            ids = [tst_id]
    else:
        ids = tst_id

    for t_id in ids:
        f = open(path + "%02d" % t_id + "/angles.txt")
        lines = f.readlines()
        f.close()
        # frames = np.zeros((len(lines)))
        # angles = np.zeros((len(lines), 3))
        for i, l in enumerate(lines):
            val = l[:-1].split("\t")
            # frames[i] = val[0]

            if filter_angles:
                roll = float(val[1])
                pitch = float(val[2])
                yaw = float(val[3])
                if abs(yaw) < 40 or abs(pitch) < 20:
                    continue

            data.append(
                {'id': t_id, 'frame': val[0], 'angle1': float(val[1]), 'angle2': float(val[2]), 'angle3': float(val[3])})
    return data


def pre_load_val(path, val_id=[11,12]):
    return pre_load(path, val_id)


def load_data(data, pathX, img_rows=224, img_cols=224, channels=3,
              standardize=False,
              contrast=False,
              equalize=False,
              stretch_img=False,
              data_augmentation=True,
              angle_norm_method=1,
              max_angle=90.0):

    X = np.zeros((len(data), channels, img_rows, img_cols))
    Y = np.zeros((len(data), 3))

    for i, d in enumerate(data):
        folder = "%02d/" % d['id'] + "/"
        file = "frame_" + d['frame'] + "_face_depth.png"

        if channels > 1:
            img = cv2.imread(pathX + folder + file, 1)
        else:
            img = cv2.imread(pathX + folder + file, 0)

        if img is None:
            continue

        if data_augmentation:
            data_augmentation_idx = d['data_augmentation']
            rdim = random.randrange(1, 15)

            # Corner Top-left
            if data_augmentation_idx == 1:
                img = img[:img.shape[0]-rdim, :img.shape[1]-rdim]
            # Corner Top-right
            elif data_augmentation_idx == 2:
                img = img[:img.shape[0]-rdim, rdim:img.shape[1]]
            # Corner Bottom-left
            elif data_augmentation_idx == 3:
                img = img[rdim:img.shape[0], :img.shape[1]-rdim]
            # Corner bottom-right
            elif data_augmentation_idx == 4:
                img = img[rdim:img.shape[0], rdim:img.shape[1]]
            # Center
            elif data_augmentation_idx == 5:
                img = img[int(rdim/2):img.shape[0]-int(rdim/2), int(rdim/2):img.shape[1]-int(rdim/2)]
            # Top
            elif data_augmentation_idx == 6:
                img = img[int(rdim/2):, :]
            # Bottom
            elif data_augmentation_idx == 7:
                img = img[:img.shape[0]-int(rdim/2), :]
            # Left
            elif data_augmentation_idx == 8:
                img = img[:, int(rdim/2):]
            # Right
            elif data_augmentation_idx == 9:
                img = img[:, :img.shape[1]-int(rdim/2)]
            elif data_augmentation_idx == 10:
                noise = np.zeros_like(img, 'float') + np.random.normal(0.0, 5, img.shape)
                img = (img + noise).astype(np.uint8)
                img[img < 0] = 0
                img[img > 255] = 255


        if stretch_img:
            img = stretch(img, 190)

        if equalize:
            cv2.equalizeHist(img.astype(np.uint8), img)

        if contrast:
            img = img.astype(np.float32)
            if channels > 1:
                for i_ch in range(channels):
                    img[..., i_ch] = exposure.rescale_intensity(img[...,i_ch].astype('float'), out_range=(0, 1))
            else:
                p2, p98 = np.percentile(img[img < 100], (2, 98))
                img = exposure.rescale_intensity(img.astype('float'), in_range=(p2, p98), out_range=(0,1))

        if standardize:
            img = img.astype(np.float32)
            if channels > 1:
                for i_ch in range(channels):
                    img[...,i_ch] = preprocessing.scale(img[...,i_ch].astype('float'))
            else:
                img = preprocessing.scale(img.astype('float'))

        img = img.astype(np.float32)


        if img.shape[1] != img_cols or img.shape[0] != img_rows:
            img = cv2.resize(img, (img_cols,img_rows))


        if channels == 1:
            img = np.expand_dims(img, axis=0)
        else:
            img = np.reshape(img, (img.shape[2], img.shape[0], img.shape[1]))


        X[i] = img

        if angle_norm_method == 1:
            Y[i, 0] = (d['angle1'] + max_angle) / (2.0*max_angle)
            Y[i, 1] = (d['angle2'] + max_angle) / (2.0*max_angle)
            Y[i, 2] = (d['angle3'] + max_angle) / (2.0*max_angle)
        elif angle_norm_method == 0:
            Y[i, 0] = (d['angle1'])
            Y[i, 1] = (d['angle2'])
            Y[i, 2] = (d['angle3'])
        elif angle_norm_method == -1:
            Y[i, 0] = d['angle1'] / float(max_angle)
            Y[i, 1] = d['angle2'] / float(max_angle)
            Y[i, 2] = d['angle3'] / float(max_angle)
        else:
            raise ValueError('Flag: "angle_norm_method" is out of range.')

    return X, Y


def load_data_val(data, pathX, img_rows=224, img_cols=224, channels=3,
              standardize=False,
              contrast=False,
              equalize=False,
              stretch_img=False,
              data_augmentation=False,
              angle_norm_method=1,
              OF = False,
              max_angle=90.0):

    X = np.zeros((len(data), channels, img_rows, img_cols))
    Y = np.zeros((len(data), 3))

    last_id = 0
    last_frame = 0
    ridx = 0
    for i, d in enumerate(data):
        folder = "%02d/" % d['id'] + "/"
        file = "frame_" + d['frame'] + "_face_depth.png"

        if channels == 1:
            img = cv2.imread(pathX + folder + file, 0)
        else:
            img = cv2.imread(pathX + folder + file, 1)

        if img is None:
            continue

        if data_augmentation:
            if last_id == int(d['id']) and last_frame == int(d['frame']):
                ridx += 1
            else:
                ridx = 0
            rdim = 14

            if ridx == 1:
                img = img[:img.shape[0]-rdim, :img.shape[1]-rdim]
            elif ridx == 2:
                img = img[:img.shape[0]-rdim, rdim:img.shape[1]]
            elif ridx == 3:
                img = img[rdim:img.shape[0], :img.shape[1]-rdim]
            elif ridx == 4:
                img = img[rdim:img.shape[0], rdim:img.shape[1]]
            elif ridx == 5:
                img = img[int(rdim/2):img.shape[0]-int(rdim/2), int(rdim/2):img.shape[1]-int(rdim/2)]

        if stretch_img:
            img = stretch(img, 190)

        if equalize:
            cv2.equalizeHist(img.astype(np.uint8), img)

        if contrast:
            img = img.astype(np.float32)
            if channels > 1:
                for i_ch in range(channels):
                    p2, p98 = np.percentile(img[..., i_ch][img[..., i_ch] < 100], (2, 98))
                    img[..., i_ch] = exposure.rescale_intensity(img[...,i_ch].astype('float'), in_range=(p2, p98), out_range=(0, 1))
            else:
                p2, p98 = np.percentile(img[img < 100], (2, 98))
                img = exposure.rescale_intensity(img.astype('float'), in_range=(p2, p98), out_range=(0,1))

        if standardize:
            img = img.astype(np.float32)
            if channels > 1:
                for i_ch in range(channels):
                    img[...,i_ch] = preprocessing.scale(img[...,i_ch].astype('float'))
            else:
                img = preprocessing.scale(img.astype('float'))

        img = img.astype(np.float32)

        if img.shape[1] != img_cols or img.shape[0] != img_rows:
            img = cv2.resize(img, (img_cols,img_rows))

        if channels == 1:
            img = np.expand_dims(img, axis=0)
        else:
            img = np.reshape(img, (img.shape[2], img.shape[0], img.shape[1]))

        X[i] = img

        if angle_norm_method == 1:
            Y[i, 0] = (d['angle1'] + max_angle) / (2.0*max_angle)
            Y[i, 1] = (d['angle2'] + max_angle) / (2.0*max_angle)
            Y[i, 2] = (d['angle3'] + max_angle) / (2.0*max_angle)
        elif angle_norm_method == 0:
            Y[i, 0] = (d['angle1'])
            Y[i, 1] = (d['angle2'])
            Y[i, 2] = (d['angle3'])
        elif angle_norm_method == -1:
            Y[i, 0] = d['angle1'] / float(max_angle)
            Y[i, 1] = d['angle2'] / float(max_angle)
            Y[i, 2] = d['angle3'] / float(max_angle)
        else:
            raise ValueError('Flag: "angle_norm_method" is out of range.')

        last_id = int(d['id'])
        last_frame = int(d['frame'])

    return X, Y


def load_data_OF(data, pathX, img_rows=224, img_cols=224, channels=3,
                selected_channels=[0,1,2,3],
                file_name_suffix=None,
                standardize=False,
                contrast=False,
                equalize=False,
                stretch_img=False,
                data_augmentation=True,
                angle_norm_method=1,
                max_angle=90.0):

    assert (channels == len(selected_channels)), "number of channels does not agree with selected_channels parameters"

    X = np.zeros((len(data), channels, img_rows, img_cols))
    Y = np.zeros((len(data), 3))

    for i, d in enumerate(data):
        folder = "%02d/" % d['id'] + "/"
        if file_name_suffix is None:
            file = "frame_" + d['frame'] + "_face_OF.png"
        else:
            file = "frame_" + d['frame'] + file_name_suffix + ".png"

        img = cv2.imread(pathX + folder + file, -1)

        if img is None:
            continue

        img = img[..., selected_channels]


        if data_augmentation:
            data_augmentation_idx = d['data_augmentation']
            rdim = random.randrange(1, 15)

            # Corner Top-left
            if data_augmentation_idx == 1:
                img = img[:img.shape[0] - rdim, :img.shape[1] - rdim]
            # Corner Top-right
            elif data_augmentation_idx == 2:
                img = img[:img.shape[0] - rdim, rdim:img.shape[1]]
            # Corner Bottom-left
            elif data_augmentation_idx == 3:
                img = img[rdim:img.shape[0], :img.shape[1] - rdim]
            # Corner bottom-right
            elif data_augmentation_idx == 4:
                img = img[rdim:img.shape[0], rdim:img.shape[1]]
            # Center
            elif data_augmentation_idx == 5:
                img = img[int(rdim / 2):img.shape[0] - int(rdim / 2), int(rdim / 2):img.shape[1] - int(rdim / 2)]
            # Top
            elif data_augmentation_idx == 6:
                img = img[int(rdim / 2):, :]
            # Bottom
            elif data_augmentation_idx == 7:
                img = img[:img.shape[0] - int(rdim / 2), :]
            # Left
            elif data_augmentation_idx == 8:
                img = img[:, int(rdim / 2):]
            # Right
            elif data_augmentation_idx == 9:
                img = img[:, :img.shape[1] - int(rdim / 2)]
            elif data_augmentation_idx == 10:
                noise = np.zeros_like(img, 'float') + np.random.normal(0.0, 5, img.shape)
                img = (img + noise).astype(np.uint8)
                img[img < 0] = 0
                img[img > 255] = 255

        if stretch_img:
            img = stretch(img, 190)

        if equalize:
            cv2.equalizeHist(img.astype(np.uint8), img)

        if contrast:
            img = img.astype(np.float32)
            if channels > 1 and selected_channels != [0,1]:
                for i_ch in range(channels):
                    img[..., i_ch] = exposure.rescale_intensity(img[..., i_ch].astype('float'), out_range=(0, 1))
            else:
                img = exposure.rescale_intensity(img.astype('float'), out_range=(0, 1))

        if standardize:
            img = img.astype(np.float32)
            if channels > 1:
                for i_ch in range(channels):
                    img[..., i_ch] = preprocessing.scale(img[..., i_ch].astype('float'))
            else:
                img = preprocessing.scale(img.astype('float'))

        img = img.astype(np.float32)

        if img.shape[1] != img_cols or img.shape[0] != img_rows:
            img = cv2.resize(img, (img_cols, img_rows))

        if channels == 1:
            img = np.expand_dims(img, axis=0)
        else:
            img = np.reshape(img, (img.shape[2], img.shape[0], img.shape[1]))

        X[i] = img

        if angle_norm_method == 1:
            Y[i, 0] = (d['angle1'] + max_angle) / (2.0 * max_angle)
            Y[i, 1] = (d['angle2'] + max_angle) / (2.0 * max_angle)
            Y[i, 2] = (d['angle3'] + max_angle) / (2.0 * max_angle)
        elif angle_norm_method == 0:
            Y[i, 0] = (d['angle1'])
            Y[i, 1] = (d['angle2'])
            Y[i, 2] = (d['angle3'])
        elif angle_norm_method == -1:
            Y[i, 0] = d['angle1'] / float(max_angle)
            Y[i, 1] = d['angle2'] / float(max_angle)
            Y[i, 2] = d['angle3'] / float(max_angle)
        else:
            raise ValueError('Flag: "angle_norm_method" is out of range.')

    return X, Y


def load_data_OF_new(data, pathX, img_rows=224, img_cols=224, channels=3,
                selected_channels=[0,1,2,3],
                file_name_suffix=None,
                standardize=False,
                contrast=False,
                equalize=False,
                stretch_img=False,
                data_augmentation=True,
                angle_norm_method=1,
                max_angle=90.0):

    assert (channels == len(selected_channels)), "number of channels does not agree with selected_channels parameters"

    X = np.zeros((len(data), channels, img_rows, img_cols))
    Y = np.zeros((len(data), 3))

    for i, d in enumerate(data):
        folder = "%02d/" % d['id'] + "/"
        if file_name_suffix is None:
            file = "frame_" + d['frame'] + "_face_OF.png"
        else:
            file = "frame_" + d['frame'] + file_name_suffix + ".png"

        img = cv2.imread(pathX + folder + file, -1)

        if img is None:
            continue

        img = img[..., selected_channels]

        if data_augmentation:
            data_augmentation_idx = d['data_augmentation']
            rdim = random.randrange(1, 15)

            # Corner Top-left
            if data_augmentation_idx == 1:
                img = img[:img.shape[0] - rdim, :img.shape[1] - rdim]
            # Corner Top-right
            elif data_augmentation_idx == 2:
                img = img[:img.shape[0] - rdim, rdim:img.shape[1]]
            # Corner Bottom-left
            elif data_augmentation_idx == 3:
                img = img[rdim:img.shape[0], :img.shape[1] - rdim]
            # Corner bottom-right
            elif data_augmentation_idx == 4:
                img = img[rdim:img.shape[0], rdim:img.shape[1]]
            # Center
            elif data_augmentation_idx == 5:
                img = img[int(rdim / 2):img.shape[0] - int(rdim / 2), int(rdim / 2):img.shape[1] - int(rdim / 2)]
            # Top
            elif data_augmentation_idx == 6:
                img = img[int(rdim / 2):, :]
            # Bottom
            elif data_augmentation_idx == 7:
                img = img[:img.shape[0] - int(rdim / 2), :]
            # Left
            elif data_augmentation_idx == 8:
                img = img[:, int(rdim / 2):]
            # Right
            elif data_augmentation_idx == 9:
                img = img[:, :img.shape[1] - int(rdim / 2)]
            elif data_augmentation_idx == 10:
                noise = np.zeros_like(img, 'float') + np.random.normal(0.0, 5, img.shape)
                img = (img + noise).astype(np.uint8)
                img[img < 0] = 0
                img[img > 255] = 255

        if stretch_img:
            img = stretch(img, 190)

        if equalize:
            cv2.equalizeHist(img.astype(np.uint8), img)

        if contrast:
            img = img.astype(np.float32)
            if channels > 1 and selected_channels != [0,1]:
                for i_ch in range(channels):
                    img[..., i_ch] = exposure.rescale_intensity(img[..., i_ch].astype('float'), out_range=(0, 1))
            else:
                img = exposure.rescale_intensity(img.astype('float'), out_range=(0, 1))

        if standardize:
            img = img.astype(np.float32)
            if channels > 1:
                for i_ch in range(channels):
                    img[..., i_ch] = preprocessing.scale(img[..., i_ch].astype('float'))
            else:
                img = preprocessing.scale(img.astype('float'))

        img = img.astype(np.float32)

        if img.shape[1] != img_cols or img.shape[0] != img_rows:
            img = cv2.resize(img, (img_cols, img_rows))

        if channels == 1:
            img = np.expand_dims(img, axis=0)
        else:
            img = np.transpose(img, (2, 0, 1))

        X[i] = img

        if angle_norm_method == 1:
            Y[i, 0] = (d['angle1'] + max_angle) / (2.0 * max_angle)
            Y[i, 1] = (d['angle2'] + max_angle) / (2.0 * max_angle)
            Y[i, 2] = (d['angle3'] + max_angle) / (2.0 * max_angle)
        elif angle_norm_method == 0:
            Y[i, 0] = (d['angle1'])
            Y[i, 1] = (d['angle2'])
            Y[i, 2] = (d['angle3'])
        elif angle_norm_method == -1:
            Y[i, 0] = d['angle1'] / float(max_angle)
            Y[i, 1] = d['angle2'] / float(max_angle)
            Y[i, 2] = d['angle3'] / float(max_angle)
        else:
            raise ValueError('Flag: "angle_norm_method" is out of range.')

    return X, Y


def load_data_val_OF_new(data, pathX, img_rows=224, img_cols=224, channels=4,
                      selected_channels=[0,1,2,3],
                        file_name_suffix=None,
                      standardize=False,
                      normalize=True,
                      contrast=False,
                      equalize=False,
                      stretch_img=False,
                      data_augmentation=False,
                      angle_norm_method=1,
                      max_angle=90.0):

    assert (channels == len(selected_channels)), "number of channels does not agree with selected_channels parameters"

    X = np.zeros((len(data), channels, img_rows, img_cols))
    Y = np.zeros((len(data), 3))

    last_id = 0
    last_frame = 0
    ridx = 0
    for i, d in enumerate(data):
        folder = "%02d/" % d['id'] + "/"
        if file_name_suffix is None:
            file = "frame_" + d['frame'] + "_face_OF.png"
        else:
            file = "frame_" + d['frame'] + file_name_suffix + ".png"
        img = cv2.imread(pathX + folder + file, -1)

        if img is None:
            continue

        img = img[..., selected_channels]

        if data_augmentation:
            if last_id == int(d['id']) and last_frame == int(d['frame']):
                ridx += 1
            else:
                ridx = 0
            rdim = 14

            if ridx == 1:
                img = img[:img.shape[0] - rdim, :img.shape[1] - rdim]
            elif ridx == 2:
                img = img[:img.shape[0] - rdim, rdim:img.shape[1]]
            elif ridx == 3:
                img = img[rdim:img.shape[0], :img.shape[1] - rdim]
            elif ridx == 4:
                img = img[rdim:img.shape[0], rdim:img.shape[1]]
            elif ridx == 5:
                img = img[int(rdim / 2):img.shape[0] - int(rdim / 2), int(rdim / 2):img.shape[1] - int(rdim / 2)]

        # cv2.imshow("", img)
        # cv2.waitKey()

        if stretch_img:
            img = stretch(img, 190)

        if equalize:
            cv2.equalizeHist(img.astype(np.uint8), img)

        if contrast:
            img = img.astype(np.float32)
            if channels > 1 and selected_channels != [0,1]:
                for i_ch in range(channels):
                    img[..., i_ch] = exposure.rescale_intensity(img[..., i_ch].astype('float'), out_range=(0, 1))
            else:
                img = exposure.rescale_intensity(img.astype('float'), out_range=(0, 1))

        if standardize:
            img = img.astype(np.float32)
            # img = cv2.normalize(img.astype(np.float32), None, 0, 1, cv2.NORM_MINMAX)
            if channels > 1:
                for i_ch in range(channels):
                    img[..., i_ch] = preprocessing.scale(img[..., i_ch].astype('float'))
            else:
                img = preprocessing.scale(img.astype('float'))

        img = img.astype(np.float32)

        # cv2.imshow("", img)
        # cv2.waitKey()

        if img.shape[1] != img_cols or img.shape[0] != img_rows:
            img = cv2.resize(img, (img_cols, img_rows))

        if channels == 1:
            img = np.expand_dims(img, axis=0)
        else:
            # img = np.reshape(img, (img.shape[2], img.shape[0], img.shape[1]))
            img = np.transpose(img, (2, 0, 1))

        X[i] = img

        if angle_norm_method == 1:
            Y[i, 0] = (d['angle1'] + max_angle) / (2.0 * max_angle)
            Y[i, 1] = (d['angle2'] + max_angle) / (2.0 * max_angle)
            Y[i, 2] = (d['angle3'] + max_angle) / (2.0 * max_angle)
            # Y[i, 0] = (d['angle1'] + 180.0) / 360.0
            # Y[i, 1] = (d['angle2'] + 180.0) / 360.0
            # Y[i, 2] = (d['angle3'] + 180.0) / 360.0
        elif angle_norm_method == 0:
            Y[i, 0] = (d['angle1'])
            Y[i, 1] = (d['angle2'])
            Y[i, 2] = (d['angle3'])
        elif angle_norm_method == -1:
            Y[i, 0] = d['angle1'] / float(max_angle)
            Y[i, 1] = d['angle2'] / float(max_angle)
            Y[i, 2] = d['angle3'] / float(max_angle)
        else:
            raise ValueError('Flag: "angle_norm_method" is out of range.')

        last_id = int(d['id'])
        last_frame = int(d['frame'])

    return X, Y

	
def load_data_gray(data, pathX, img_rows=224, img_cols=224, channels=3,
                file_name_suffix=None,
                standardize=False,
                contrast=False,
                equalize=False,
                stretch_img=False,
                data_augmentation=True,
                angle_norm_method=1,
                max_angle=90.0):

    X = np.zeros((len(data), channels, img_rows, img_cols))
    Y = np.zeros((len(data), 3))

    for i, d in enumerate(data):
        folder = "%02d/" % d['id'] + "/"
        if file_name_suffix is None:
            file = "frame_" + d['frame'] + "_face_gray.png"
        else:
            file = "frame_" + d['frame'] + file_name_suffix + ".png"

        if channels > 1:
            img = cv2.imread(pathX + folder + file, 1)
        else:
            img = cv2.imread(pathX + folder + file, 0)

        if img is None:
            continue


        if data_augmentation:
            data_augmentation_idx = d['data_augmentation']
            rdim = random.randrange(1, 15)

            # Corner Top-left
            if data_augmentation_idx == 1:
                img = img[:img.shape[0] - rdim, :img.shape[1] - rdim]
            # Corner Top-right
            elif data_augmentation_idx == 2:
                img = img[:img.shape[0] - rdim, rdim:img.shape[1]]
            # Corner Bottom-left
            elif data_augmentation_idx == 3:
                img = img[rdim:img.shape[0], :img.shape[1] - rdim]
            # Corner bottom-right
            elif data_augmentation_idx == 4:
                img = img[rdim:img.shape[0], rdim:img.shape[1]]
            # Center
            elif data_augmentation_idx == 5:
                img = img[int(rdim / 2):img.shape[0] - int(rdim / 2), int(rdim / 2):img.shape[1] - int(rdim / 2)]
            # Top
            elif data_augmentation_idx == 6:
                img = img[int(rdim / 2):, :]
            # Bottom
            elif data_augmentation_idx == 7:
                img = img[:img.shape[0] - int(rdim / 2), :]
            # Left
            elif data_augmentation_idx == 8:
                img = img[:, int(rdim / 2):]
            # Right
            elif data_augmentation_idx == 9:
                img = img[:, :img.shape[1] - int(rdim / 2)]
            elif data_augmentation_idx == 10:
                noise = np.zeros_like(img, 'float') + np.random.normal(0.0, 5, img.shape)
                img = (img + noise).astype(np.uint8)
                img[img < 0] = 0
                img[img > 255] = 255


        if stretch_img:
            img = stretch(img, 190)

        if equalize:
            cv2.equalizeHist(img.astype(np.uint8), img)

        if contrast:
            img = img.astype(np.float32)
            if channels > 1:
                for i_ch in range(channels):
                    img[..., i_ch] = exposure.rescale_intensity(img[..., i_ch].astype('float'), out_range=(0, 1))
            else:
                img = exposure.rescale_intensity(img.astype('float'), out_range=(0, 1))

        if standardize:
            img = img.astype(np.float32)
            if channels > 1:
                for i_ch in range(channels):
                    img[..., i_ch] = preprocessing.scale(img[..., i_ch].astype('float'))
            else:
                img = preprocessing.scale(img.astype('float'))

        img = img.astype(np.float32)

        if img.shape[1] != img_cols or img.shape[0] != img_rows:
            img = cv2.resize(img, (img_cols, img_rows))

        if channels == 1:
            img = np.expand_dims(img, axis=0)
        else:
            img = np.reshape(img, (img.shape[2], img.shape[0], img.shape[1]))

        X[i] = img

        if angle_norm_method == 1:
            Y[i, 0] = (d['angle1'] + max_angle) / (2.0 * max_angle)
            Y[i, 1] = (d['angle2'] + max_angle) / (2.0 * max_angle)
            Y[i, 2] = (d['angle3'] + max_angle) / (2.0 * max_angle)
        elif angle_norm_method == 0:
            Y[i, 0] = (d['angle1'])
            Y[i, 1] = (d['angle2'])
            Y[i, 2] = (d['angle3'])
        elif angle_norm_method == -1:
            Y[i, 0] = d['angle1'] / float(max_angle)
            Y[i, 1] = d['angle2'] / float(max_angle)
            Y[i, 2] = d['angle3'] / float(max_angle)
        else:
            raise ValueError('Flag: "angle_norm_method" is out of range.')

    return X, Y


def load_data_val_gray(data, pathX, img_rows=224, img_cols=224, channels=4,
                        file_name_suffix=None,
                      standardize=False,
                      normalize=True,
                      contrast=False,
                      equalize=False,
                      stretch_img=False,
                      data_augmentation=False,
                      angle_norm_method=1,
                      max_angle=90.0):

    X = np.zeros((len(data), channels, img_rows, img_cols))
    Y = np.zeros((len(data), 3))

    last_id = 0
    last_frame = 0
    ridx = 0
    for i, d in enumerate(data):
        folder = "%02d/" % d['id'] + "/"
        if file_name_suffix is None:
            file = "frame_" + d['frame'] + "_face_gray.png"
        else:
            file = "frame_" + d['frame'] + file_name_suffix + ".png"

        if channels > 1:
            img = cv2.imread(pathX + folder + file, 1)
        else:
            img = cv2.imread(pathX + folder + file, 0)

        if img is None:
            continue

        if data_augmentation:
            if last_id == int(d['id']) and last_frame == int(d['frame']):
                ridx += 1
            else:
                ridx = 0
            rdim = 14

            if ridx == 1:
                img = img[:img.shape[0] - rdim, :img.shape[1] - rdim]
            elif ridx == 2:
                img = img[:img.shape[0] - rdim, rdim:img.shape[1]]
            elif ridx == 3:
                img = img[rdim:img.shape[0], :img.shape[1] - rdim]
            elif ridx == 4:
                img = img[rdim:img.shape[0], rdim:img.shape[1]]
            elif ridx == 5:
                img = img[int(rdim / 2):img.shape[0] - int(rdim / 2), int(rdim / 2):img.shape[1] - int(rdim / 2)]

        if stretch_img:
            img = stretch(img, 190)

        if equalize:
            cv2.equalizeHist(img.astype(np.uint8), img)

        if contrast:
            img = img.astype(np.float32)
            if channels > 1:
                for i_ch in range(channels):
                    img[..., i_ch] = exposure.rescale_intensity(img[..., i_ch].astype('float'), out_range=(0, 1))
            else:
                img = exposure.rescale_intensity(img.astype('float'), out_range=(0, 1))

        if standardize:
            img = img.astype(np.float32)
            if channels > 1:
                for i_ch in range(channels):
                    img[..., i_ch] = preprocessing.scale(img[..., i_ch].astype('float'))
            else:
                img = preprocessing.scale(img.astype('float'))

        img = img.astype(np.float32)

        if img.shape[1] != img_cols or img.shape[0] != img_rows:
            img = cv2.resize(img, (img_cols, img_rows))

        if channels == 1:
            img = np.expand_dims(img, axis=0)
        else:
            img = np.reshape(img, (img.shape[2], img.shape[0], img.shape[1]))

        X[i] = img

        if angle_norm_method == 1:
            Y[i, 0] = (d['angle1'] + max_angle) / (2.0 * max_angle)
            Y[i, 1] = (d['angle2'] + max_angle) / (2.0 * max_angle)
            Y[i, 2] = (d['angle3'] + max_angle) / (2.0 * max_angle)
        elif angle_norm_method == 0:
            Y[i, 0] = (d['angle1'])
            Y[i, 1] = (d['angle2'])
            Y[i, 2] = (d['angle3'])
        elif angle_norm_method == -1:
            Y[i, 0] = d['angle1'] / float(max_angle)
            Y[i, 1] = d['angle2'] / float(max_angle)
            Y[i, 2] = d['angle3'] / float(max_angle)
        else:
            raise ValueError('Flag: "angle_norm_method" is out of range.')

        last_id = int(d['id'])
        last_frame = int(d['frame'])

    return X, Y