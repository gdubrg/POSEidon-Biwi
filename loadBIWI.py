from __future__ import division, print_function
import numpy as np
import cv2
from rowStretching import stretch
from sklearn import preprocessing
from skimage import exposure, util
import random

# def crop(img, rows=224, cols=224):
# 	img_crop = []
# 	if img.shape[0] < img.shape[1]:
# 		if img.shape[0] != rows:
# 			new_cols = int((img.shape[1]*rows)/img.shape[0])
# 			img = cv2.resize(img, (new_cols, rows))
# 		e1 = int((img.shape[1] - cols)/2)
# 		e2 = e1 + cols
# 		img_crop = img[:, e1:e2]
#
# 	if img.shape[0] > img.shape[1]:
# 		if img.shape[1] != cols:
# 			new_rows = int((img.shape[0]*cols)/img.shape[1])
# 			img = cv2.resize(img, (cols, new_rows))
# 		e1 = int((img.shape[0] - rows)/2)
# 		e2 = e1 + rows
# 		img_crop = img[e1:e2, :]
#
# 	if img.shape[0] == img.shape[1]:
# 		if img.shape[1] != cols:
# 			img = cv2.resize(img, (cols, rows))
# 		img_crop = img
#
# 	return img_crop

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


def load_data_random(data, pathX, img_rows=224, img_cols=224, channels=3,
              standardize=False,
              contrast=False,
              equalize=False,
              stretch_img=False,
              data_augmentation=False,
              angle_norm_method=1,
              max_angle=90.0):

    X = np.zeros((len(data), channels, img_rows, img_cols))
    Y = np.zeros((len(data), 3))

    for i, d in enumerate(data):
        folder = "%02d/" % d['id'] + "/"
        file = "frame_" + d['frame'] + "_face_depth.png"
        img = cv2.imread(pathX + folder + file, 0)

        if img is None:
            continue

        if data_augmentation:
            ridx = random.randrange(0,6)
            rdim = random.randrange(0,14)

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

        # cv2.imshow("", img)
        # cv2.waitKey()

        if stretch_img:
            img = stretch(img, 190)

        if equalize:
            cv2.equalizeHist(img.astype(np.uint8), img)

        if contrast:
            p2, p98 = np.percentile(img[img < 100], (2, 98))
            img = exposure.rescale_intensity(img.astype('float'), in_range=(p2, p98), out_range=(0,1))

        ###############################################################################################################
        # img = (img*255).astype(np.uint8)
        #
        # img_RGB = cv2.imread("../../Dataset/BIWI/face_dataset_RGB3/" + folder + file[:-10] + "_rgb.png", 0)
        # p2, p98 = np.percentile(img_RGB, (2, 98))
        # img_RGB = exposure.rescale_intensity(img_RGB.astype('float'), in_range=(p2, p98), out_range=(0,255))
        # img_RGB = img_RGB.astype(np.uint8)
        #
        # alpha = 0.2
        # img_blend = (alpha * img_RGB + (1-alpha) * img).astype(np.uint8)
        #
        # cv2.imshow("DEPTH", img)
        # cv2.imshow("RGB", img_RGB)
        # cv2.imshow("BLEND", img_blend)
        # cv2.waitKey()
        ###############################################################################################################

        if standardize:
            # img = cv2.normalize(img.astype(np.float32), None, 0, 1, cv2.NORM_MINMAX)
            img = preprocessing.scale(img.astype('float'))

        # if normalize:
        #     _, mask = cv2.threshold(img, 190, 1, cv2.THRESH_BINARY_INV)
        #     img = img * mask
        #     img = cv2.normalize(img.astype(np.float32), None, 0, 1, cv2.NORM_MINMAX)
        # elif equalize:
        #     _, mask = cv2.threshold(img, 190, 1, cv2.THRESH_BINARY_INV)
        #     img = img * mask
        #     # img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        #     cv2.equalizeHist(img.astype(np.uint8), img)
        #     img = cv2.normalize(img.astype(np.float32), None, 0, 1, cv2.NORM_MINMAX)

        img = img.astype(np.float32)

        # cv2.imshow("", img)
        # cv2.waitKey()

        if img.shape[1] != img_cols or img.shape[0] != img_rows:
            img = cv2.resize(img, (img_cols,img_rows))

        img = np.expand_dims(img, axis=0)

        if channels > 1:
            img = np.repeat(img, channels, axis=0)

        # img[0, :, :] -= 103.939
        # img[1, :, :] -= 116.779
        # img[2, :, :] -= 123.68

        X[i] = img

        if angle_norm_method == 1:
            Y[i, 0] = (d['angle1'] + max_angle) / (2.0*max_angle)
            Y[i, 1] = (d['angle2'] + max_angle) / (2.0*max_angle)
            Y[i, 2] = (d['angle3'] + max_angle) / (2.0*max_angle)
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

    return X, Y


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

        # cv2.imshow("", img)
        # cv2.waitKey()

        if data_augmentation:
            data_augmentation_idx = d['data_augmentation']
            # TODO: remove
            # rdim = random.randrange(5, 50)
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
                # noise = 0.4 * img.std() * np.random.random(img.shape)
                img = (img + noise).astype(np.uint8)
                img[img < 0] = 0
                img[img > 255] = 255
                # img = (util.random_noise(img, mode='gaussian', mean=0.0, var=0.001) * 255).astype(np.uint8)

            # cv2.imshow("aug", img)
            # cv2.waitKey()

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

        ###############################################################################################################
        # img = (img*255).astype(np.uint8)
        #
        # img_RGB = cv2.imread("../../Dataset/BIWI/face_dataset_RGB3/" + folder + file[:-10] + "_rgb.png", 0)
        # p2, p98 = np.percentile(img_RGB, (2, 98))
        # img_RGB = exposure.rescale_intensity(img_RGB.astype('float'), in_range=(p2, p98), out_range=(0,255))
        # img_RGB = img_RGB.astype(np.uint8)
        #
        # alpha = 0.2
        # img_blend = (alpha * img_RGB + (1-alpha) * img).astype(np.uint8)
        #
        # cv2.imshow("DEPTH", img)
        # cv2.imshow("RGB", img_RGB)
        # cv2.imshow("BLEND", img_blend)
        # cv2.waitKey()
        ###############################################################################################################

        if standardize:
            img = img.astype(np.float32)
            if channels > 1:
                for i_ch in range(channels):
                    img[...,i_ch] = preprocessing.scale(img[...,i_ch].astype('float'))
            else:
                img = preprocessing.scale(img.astype('float'))

        # if normalize:
        #     _, mask = cv2.threshold(img, 190, 1, cv2.THRESH_BINARY_INV)
        #     img = img * mask
        #     img = cv2.normalize(img.astype(np.float32), None, 0, 1, cv2.NORM_MINMAX)
        # elif equalize:
        #     _, mask = cv2.threshold(img, 190, 1, cv2.THRESH_BINARY_INV)
        #     img = img * mask
        #     # img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        #     cv2.equalizeHist(img.astype(np.uint8), img)
        #     img = cv2.normalize(img.astype(np.float32), None, 0, 1, cv2.NORM_MINMAX)

        img = img.astype(np.float32)

        # cv2.imshow("", img)
        # cv2.waitKey()

        if img.shape[1] != img_cols or img.shape[0] != img_rows:
            img = cv2.resize(img, (img_cols,img_rows))

        # img = np.expand_dims(img, axis=0)
        # if channels > 1:
        #     img = np.repeat(img, channels, axis=0)

        if channels == 1:
            img = np.expand_dims(img, axis=0)
        else:
            img = np.reshape(img, (img.shape[2], img.shape[0], img.shape[1]))

        # img[0, :, :] -= 103.939
        # img[1, :, :] -= 116.779
        # img[2, :, :] -= 123.68

        X[i] = img

        if angle_norm_method == 1:
            Y[i, 0] = (d['angle1'] + max_angle) / (2.0*max_angle)
            Y[i, 1] = (d['angle2'] + max_angle) / (2.0*max_angle)
            Y[i, 2] = (d['angle3'] + max_angle) / (2.0*max_angle)
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

        # cv2.imshow("", img)
        # cv2.waitKey()

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
            # img = cv2.normalize(img.astype(np.float32), None, 0, 1, cv2.NORM_MINMAX)
            if channels > 1:
                for i_ch in range(channels):
                    img[...,i_ch] = preprocessing.scale(img[...,i_ch].astype('float'))
            else:
                img = preprocessing.scale(img.astype('float'))

        img = img.astype(np.float32)

        # cv2.imshow("", img)
        # cv2.waitKey()

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


def load_data_class(data, pathX, img_rows=224, img_cols=224, nb_class=36, channels=3, normalize=False, equalize=False):

    X = np.zeros((len(data), channels, img_rows, img_cols))
    Y1 = np.zeros((len(data), nb_class))
    Y2 = np.zeros((len(data), nb_class))
    Y3 = np.zeros((len(data), nb_class))

    for i, d in enumerate(data):
        folder = "%02d/" % d['id'] + "/"
        file = "frame_" + d['frame'] + "_face_depth.png"
        img = cv2.imread(pathX + folder + file, 0)

        if img is None:
            continue

        if normalize:
            _, mask = cv2.threshold(img, 190, 1, cv2.THRESH_BINARY_INV)
            img = img * mask
            img = cv2.normalize(img.astype(np.float32), None, 0, 1, cv2.NORM_MINMAX)
        elif equalize:
            _, mask = cv2.threshold(img, 190, 1, cv2.THRESH_BINARY_INV)
            img = img * mask
            # img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
            cv2.equalizeHist(img.astype(np.uint8), img)
            img = cv2.normalize(img.astype(np.float32), None, 0, 1, cv2.NORM_MINMAX)

        # img = img.astype(np.float32)

        # cv2.imshow("", img)
        # cv2.waitKey()


        if img.shape[1] != img_cols or img.shape[0] != img_rows:
            img = cv2.resize(img, (img_cols,img_rows))

        # cv2.imshow("", img)
        # cv2.waitKey()

        img = np.expand_dims(img, axis=0)

        if channels > 1:
            img = np.repeat(img, channels, axis=0)


        X[i] = img

        id1 = round((d['angle1']+180.0)/(360.0/nb_class))
        Y1[i, int(id1)] = 1
        id2 = round((d['angle2']+180.0)/(360.0/nb_class))
        Y2[i, int(id2)] = 1
        id3 = round((d['angle3']+180.0)/(360.0/nb_class))
        Y3[i, int(id3)] = 1


    return X, [Y1, Y2, Y3]


def save_autoencoder_predictions(val_data, pred, min=-1, path=None):
    # SALVATAGGIO PREDIZIONE
    for vd, p in zip(val_data, pred):
        vd_id = vd["id"]
        vd_frame = vd["frame"]
        p_img = p[0]
        if min < 0:
            p_img = ((p_img + 1) * 127.5)
        p_img = p_img.astype(np.uint8)
        if path is None:
            file_name = "predictions/id%02d_frame%05d.png" % (vd_id, int(vd_frame))
        else:
            file_name = path + "frame_%05d_face_gray.png" % int(vd_frame)
        cv2.imwrite(file_name, p_img)


def load_data_autoencoder(data, pathX, pathY, img_rows=224, img_cols=224, channels=3, out_img_rows=None, out_img_cols=None,
              standardize=False,
              contrast=False,
              equalize=False,
              stretch_img=False,
              data_augmentation=True):

    X = np.zeros((len(data), channels, img_rows, img_cols))
    if out_img_rows is None and out_img_cols is None:
        Y = np.zeros((len(data), channels, img_rows, img_cols))
    else:
        Y = np.zeros((len(data), channels, out_img_rows, out_img_cols))

    for i, d in enumerate(data):
        folder = "%02d/" % d['id'] + "/"
        file = "frame_" + d['frame'] + "_face_depth.png"
        fileRGB = "frame_" + d['frame'] + "_face_rgb.png"

        if channels > 1:
            imgRGB = cv2.imread(pathY + folder + fileRGB, 1)
            imgD = cv2.imread(pathX + folder + file, 1)
        else:
            imgRGB = cv2.imread(pathY + folder + fileRGB, 0)
            imgD = cv2.imread(pathX + folder + file, 0)

        if imgD is None:
            continue

        for k, img in enumerate([imgD, imgRGB]):
            if data_augmentation:
                data_augmentation_idx = d['data_augmentation']
                rdim = random.randrange(1,15)

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
                    img = cv2.flip(img, 1)
                elif data_augmentation_idx == 11:
                    rows, cols = img.shape
                    M = cv2.getRotationMatrix2D((cols/2,rows/2), rdim * 0.75, 1)
                    img = cv2.warpAffine(img,M,(cols,rows), borderMode=cv2.BORDER_REPLICATE)
                elif data_augmentation_idx == 12:
                    rows, cols = img.shape
                    M = cv2.getRotationMatrix2D((cols/2,rows/2), - rdim * 0.75, 1)
                    img = cv2.warpAffine(img,M,(cols,rows), borderMode=cv2.BORDER_REPLICATE)
                elif data_augmentation_idx == 13:
                    rows, cols = img.shape
                    M = cv2.getRotationMatrix2D((cols/2,rows/2), 0, 1 + rdim/150)
                    img = cv2.warpAffine(img,M,(cols,rows), borderMode=cv2.BORDER_REPLICATE)
                elif data_augmentation_idx == 14:
                    rows, cols = img.shape
                    M = cv2.getRotationMatrix2D((cols/2,rows/2), 0, 1 - rdim/150)
                    img = cv2.warpAffine(img,M,(cols,rows), borderMode=cv2.BORDER_REPLICATE)
                elif data_augmentation_idx == 15:
                    noise = np.zeros_like(img, 'float') + np.random.normal(0.0, 5, img.shape)
                    # noise = 0.4 * img.std() * np.random.random(img.shape)
                    img = (img + noise).astype(np.uint8)
                    img[img < 0] = 0
                    img[img > 255] = 255
                    # img = (util.random_noise(img, mode='gaussian', mean=0.0, var=0.001) * 255).astype(np.uint8)

                # cv2.imshow("", img)
                # cv2.waitKey()

            if stretch_img:
                img = stretch(img, 190)

            if equalize:
                cv2.equalizeHist(img.astype(np.uint8), img)

            if contrast:
                if k % 2 == 0:
                    p2, p98 = np.percentile(img[img < 100], (2, 98))
                    img = exposure.rescale_intensity(img.astype('float'), in_range=(p2, p98), out_range=(0,1))
                else:
                    p2, p98 = np.percentile(img, (2, 98))
                    img = exposure.rescale_intensity(img.astype('float'), in_range=(p2, p98), out_range=(-1,1))

            if standardize and k % 2 == 0:
                # img = cv2.normalize(img.astype(np.float32), None, 0, 1, cv2.NORM_MINMAX)
                if channels > 1:
                    img[..., 0] = preprocessing.scale(img[..., 0].astype('float'))
                    img[..., 1] = preprocessing.scale(img[..., 1].astype('float'))
                    img[..., 2] = preprocessing.scale(img[..., 2].astype('float'))
                else:
                    img = preprocessing.scale(img.astype('float'))

            img = img.astype(np.float32)

            # cv2.imshow("", img)
            # cv2.waitKey()

            if out_img_cols is None and out_img_rows is None:
                if img.shape[1] != img_cols or img.shape[0] != img_rows:
                    img = cv2.resize(img, (img_cols,img_rows))
            else:
                if k % 2 == 0:
                    if img.shape[1] != img_cols or img.shape[0] != img_rows:
                        img = cv2.resize(img, (img_cols,img_rows))
                else:
                    if img.shape[1] != out_img_cols or img.shape[0] != out_img_rows:
                        img = cv2.resize(img, (out_img_cols,out_img_rows))

            if channels == 1:
                img = np.expand_dims(img, axis=0)
            else:
                img = np.reshape(img, (img.shape[2], img.shape[0], img.shape[1]))

            if k % 2 == 0:
                imgD = img
            else:
                imgRGB = img

        X[i] = imgD
        Y[i] = imgRGB

    return X, Y


def load_data_val_autoencoder(data, pathX, pathY, img_rows=224, img_cols=224, channels=3, out_img_rows=None, out_img_cols=None,
              standardize=False,
              contrast=False,
              equalize=False,
              stretch_img=False,
              data_augmentation=False,
              angle_norm_method=1,
              max_angle=90.0):

    X = np.zeros((len(data), channels, img_rows, img_cols))
    if out_img_rows is None and out_img_cols is None:
        Y = np.zeros((len(data), channels, img_rows, img_cols))
    else:
        Y = np.zeros((len(data), channels, out_img_rows, out_img_cols))

    last_id = 0
    last_frame = 0
    ridx = 0
    for i, d in enumerate(data):
        folder = "%02d/" % d['id'] + "/"
        file = "frame_" + d['frame'] + "_face_depth.png"
        fileRGB = "frame_" + d['frame'] + "_face_rgb.png"

        if channels > 1:
            imgRGB = cv2.imread(pathY + folder + fileRGB, 1)
            imgD = cv2.imread(pathX + folder + file, 1)
        else:
            imgRGB = cv2.imread(pathY + folder + fileRGB, 0)
            imgD = cv2.imread(pathX + folder + file, 0)

        if imgD is None:
            continue

        for k, img in enumerate([imgD, imgRGB]):
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

            # cv2.imshow("", img)
            # cv2.waitKey()

            if stretch_img:
                img = stretch(img, 190)

            if equalize:
                cv2.equalizeHist(img.astype(np.uint8), img)

            if contrast:
                if k % 2 == 0:
                    p2, p98 = np.percentile(img[img < 100], (2, 98))
                    img = exposure.rescale_intensity(img.astype('float'), in_range=(p2, p98), out_range=(0,1))
                else:
                    p2, p98 = np.percentile(img, (2, 98))
                    img = exposure.rescale_intensity(img.astype('float'), in_range=(p2, p98), out_range=(-1,1))

            if standardize and k % 2 == 0:
                # img = cv2.normalize(img.astype(np.float32), None, 0, 1, cv2.NORM_MINMAX)
                if channels > 1:
                    img[..., 0] = preprocessing.scale(img[...,0].astype('float'))
                    img[..., 1] = preprocessing.scale(img[...,1].astype('float'))
                    img[..., 2] = preprocessing.scale(img[...,2].astype('float'))
                else:
                    img = preprocessing.scale(img.astype('float'))

            img = img.astype(np.float32)

            # cv2.imshow("", img)
            # cv2.waitKey()

            if out_img_cols is None and out_img_rows is None:
                if img.shape[1] != img_cols or img.shape[0] != img_rows:
                    img = cv2.resize(img, (img_cols,img_rows))
            else:
                if k % 2 == 0:
                    if img.shape[1] != img_cols or img.shape[0] != img_rows:
                        img = cv2.resize(img, (img_cols,img_rows))
                else:
                    if img.shape[1] != out_img_cols or img.shape[0] != out_img_rows:
                        img = cv2.resize(img, (out_img_cols,out_img_rows))

            if channels == 1:
                img = np.expand_dims(img, axis=0)
            else:
                img = np.reshape(img, (img.shape[2], img.shape[0], img.shape[1]))

            if k % 2 == 0:
                imgD = img
            else:
                imgRGB = img

        X[i] = imgD
        Y[i] = imgRGB

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

        # cv2.imshow("", img)
        # cv2.waitKey()

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
                # noise = 0.4 * img.std() * np.random.random(img.shape)
                img = (img + noise).astype(np.uint8)
                img[img < 0] = 0
                img[img > 255] = 255
                # img = (util.random_noise(img, mode='gaussian', mean=0.0, var=0.001) * 255).astype(np.uint8)

                # cv2.imshow("aug", img)
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
            img = np.reshape(img, (img.shape[2], img.shape[0], img.shape[1]))
            # img = np.transpose(img, (2, 0, 1))

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


def load_data_val_OF(data, pathX, img_rows=224, img_cols=224, channels=4,
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
            img = np.reshape(img, (img.shape[2], img.shape[0], img.shape[1]))

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

        # cv2.imshow("", img)
        # cv2.waitKey()

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
                # noise = 0.4 * img.std() * np.random.random(img.shape)
                img = (img + noise).astype(np.uint8)
                img[img < 0] = 0
                img[img > 255] = 255
                # img = (util.random_noise(img, mode='gaussian', mean=0.0, var=0.001) * 255).astype(np.uint8)

                # cv2.imshow("aug", img)
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

        # cv2.imshow("", img)
        # cv2.waitKey()

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
                # noise = 0.4 * img.std() * np.random.random(img.shape)
                img = (img + noise).astype(np.uint8)
                img[img < 0] = 0
                img[img > 255] = 255
                # img = (util.random_noise(img, mode='gaussian', mean=0.0, var=0.001) * 255).astype(np.uint8)

                # cv2.imshow("aug", img)
                # cv2.waitKey()

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

        # cv2.imshow("", img)
        # cv2.waitKey()

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

        # cv2.imshow("", img)
        # cv2.waitKey()

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
            img = np.reshape(img, (img.shape[2], img.shape[0], img.shape[1]))

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