import sys
import os
import numpy as np
from scipy import misc
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist

import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

import csv
import cv2
import spams
import time
from sklearn import svm
from sklearn.externals import joblib
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from random import shuffle, randint
import math


class Image:
    def __init__(self, color_features, magnitude_features, orientation_features, label, id, name):
        self.color_features = color_features
        self.magnitude_features = magnitude_features
        self.orientation_features = orientation_features
        self.label = label
        self.id = id
        self.name = name

class Limits:
    def __init__(self, cr_max, cg_max, cb_max, cr_min, cg_min, cb_min, g_max, g_min, o_max, o_min):
        self.cr_max = cr_max
        self.cg_max = cg_max
        self.cb_max = cb_max
        self.cr_min = cr_min
        self.cg_min = cg_min
        self.cb_min = cb_min
        self.g_max = g_max
        self.g_min = g_min
        self.o_max = o_max
        self.o_min = o_min

class Tuning:
    def __init__(self, atoms, c, g, w, accuracy, sensitivity, specitivity, cost):
        self.atoms = atoms
        self.c = c
        self.g = g
        self.w = w
        self.accuracy = accuracy
        self.sensitivity = sensitivity
        self.specitivity = specitivity
        self.cost = cost

class Result:
    def __init__(self, sensitivity, specitivity, accuracy):
        self.sensitivity = sensitivity
        self.specitivity = specitivity
        self.accuracy = accuracy

    def add(self, accuracy, sensitivity, specitivity):
        self.accuracy += accuracy
        self.sensitivity += sensitivity
        self.specitivity += specitivity

class Dictionary:
    def __init__(self, D, atoms, type, nested_no, cross_no, info):
        self.D = D
        self.atoms = atoms
        self.type = type
        self.nested_no = nested_no
        self.cross_no = cross_no
        self.info = info

class Experiment:
    def __init__(self, c_tuning_results, m_tuning_results, c_predictions, m_predictions):
        self.c_tuning_results = c_tuning_results
        self.m_tuning_results = m_tuning_results
        self.c_predictions = c_predictions
        self.m_predictions = m_predictions

def getData(desiredData, labels, train):

    rootDir = '.'
    #grid = createGrid((320, 320), 16)
    dataset = []
    replicas = []
    i = 0
    cr_max = g_max = o_max = cg_max = cb_max = 0
    cr_min = g_min = o_min = cg_min = cb_min = 100

    image_names = joblib.load('saves/edra_image_names.sav')

    #limits = Limits(cr_max, cg_max, cb_max, cr_min, cg_min, cb_min, g_max, g_min, o_max, o_min)

    #for dirName, subdirList, fileList in os.walk(rootDir, topdown=True):
    #    if dirName == './Data' + '/' + desiredData:
    #        fileList.sort()
    #        for fname in fileList:
    #            if any(fname[:(len(fname)-4)] == s for s in image_names) and ( '_O' not in fname ):
    #                image = misc.imread(dirName + '/' + fname)
    #                #print(dirName + '/' + fname)
    #                seg_path = getSegName(list(dirName + '/' + fname))
    #                #print(seg_path)
    #                binary_mask = misc.imread(seg_path)
    #                image = cropImage(image, binary_mask, 15)
    #                image = color_norm(image)
    #                limits = getLimits(image, limits)
    #                i += 1
    #                sys.stdout.write("\r{0}".format(round((float(i)/(len(fileList)/3))*100,2)))
    #                sys.stdout.flush()
    #c_max = 1.7726966281963084
    #c_min = 0.0
    #g_max = 6.696730994481612
    #g_min = 0.0
    #o_max = 1.5707963267948966
    #o_min = -1.5707963267948966
    #joblib.dump(limits, 'saves/edra_limits.sav')
    limits = joblib.load('saves/edra_limits.sav')
    i = 0
    print(limits.cr_max, limits.cg_max, limits.cb_max, limits.cr_min, limits.cg_min,
    limits.cb_min, limits.g_max, limits.g_min, limits.o_max, limits.o_min)
    for dirName, subdirList, fileList in os.walk(rootDir, topdown=True):
        if dirName == './Data' + '/' + desiredData:
            fileList.sort()
            for fname in fileList:
                if any(fname[:(len(fname)-4)] == s for s in image_names) and ( '_O' not in fname ):
                    label = whatLabel(fname[:(len(fname)-4)], labels, image_names)
                    image = misc.imread(dirName + '/' + fname)
                    seg_path = getSegName(list(dirName + '/' + fname))
                    binary_mask = misc.imread(seg_path)
                    image = cropImage(image, binary_mask, 10)
                    image = color_norm(image)
                    color_feature, magnitude_feature, orientation_feature = getFeatures(image, limits)
                    instance = Image(color_feature, magnitude_feature, orientation_feature, label, i, fname)
                    dataset.append(instance)
                    if train:
                        if label == 1:
                            for j in range(1, 2):
                                image = np.rot90(image, randint(1,3)*90)
                                #binary_mask = np.rot90(binary_mask, j)
                                color_feature, magnitude_feature, orientation_feature = getFeatures(image, limits)
                                instance = Image(color_feature, magnitude_feature, orientation_feature, label, i, fname)
                                replicas.append(instance)
                    i += 1
                    sys.stdout.write("\r{0}".format(round((float(i)/(len(fileList)/3))*100,2)))
                    sys.stdout.flush()
    shuffle(dataset)
    return dataset, replicas

def whatLabel(name, labels, list_names):

    for image, label in zip(list_names, labels):
        if name == image:
            if label == 1:
                return label
            else:
                return 0

def getSegName(path):
    new_string = None
    for i in range(0, len(path)):
        if (path[i] == '.' and path[i+1] == 'j' and path[i+2] == 'p') or (path[i] == '.' and path[i+1] == 'J' and path[i+2] == 'P'):
            new_path = path[:i] + list('_O.bmp')
            new_string = "".join(new_path)
            break
    return new_string

def getClassSamples(data):
    melanomas = []
    non_melanomas = []
    for sample in data:
        if sample.label == 1:
            melanomas.append(sample)
        else:
            non_melanomas.append(sample)

    return melanomas, non_melanomas

def getLimits(image, limits):

    current_cr_max = np.amax(image[0])
    print(';  ', current_cr_max)
    if current_cr_max > limits.cr_max:
        limits.cr_max = current_cr_max

    current_cr_min = np.amin(image[0])
    if current_cr_min < limits.cr_min:
        limits.cr_min = current_cr_min

    current_cg_max = np.amax(image[1])
    if current_cg_max > limits.cg_max:
        limits.cg_max = current_cg_max

    current_cg_min = np.amin(image[1])
    if current_cg_min < limits.cg_min:
        limits.cg_min = current_cg_min

    current_cb_max = np.amax(image[2])
    if current_cb_max > limits.cb_max:
        limits.cb_max = current_cb_max

    current_cb_min = np.amin(image[2])
    if current_cb_min < limits.cb_min:
        limits.cb_min = current_cb_min

    current_g_max, current_g_min, current_o_max, current_o_min  = getTextureHistograms(image, 16, True, limits)

    if current_g_max > limits.g_max:
        limits.g_max = current_g_max

    if current_g_min < limits.g_min:
        limits.g_min = current_g_min

    if current_o_max > limits.o_max:
        limits.o_max = current_o_max

    if current_o_min < limits.o_min:
        limits.o_min = current_o_min

    return limits

def getLabels(filename):

    label_list = []

    with open(filename) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        firstline = True
        for row in readCSV:
            if firstline:    #skip first line
                firstline = False
                continue
            label = int(float(row[1]))
            label_list.append(label)
    labels = np.array(label_list)
    return labels

def cropImage(img, binaryMask, padding):
    rect = cv2.boundingRect(binaryMask)

    if (rect[1]-padding < 0 and abs(0-(rect[1]-padding)) < padding):
        padding = abs(rect[1]-padding)
    if (rect[0]-padding < 0 and abs(0-(rect[0]-padding)) < padding):
        padding = abs(rect[0]-padding)
    if (rect[1]+rect[3]+padding > img.shape[1] and abs(img.shape[1]-(rect[1]+rect[3]+padding)) < padding):
        padding = abs(img.shape[1]-(rect[1]+rect[3]+padding))
    if (rect[0]+rect[2]+padding > img.shape[0] and abs(img.shape[0]-(rect[0]+rect[2]+padding)) < padding):
        padding = abs(img.shape[0]-(rect[0]+rect[2]+padding))

    if rect[1]-padding < 0 or rect[0]-padding < 0 or rect[1]+rect[3]+padding > img.shape[1] or rect[0]+rect[2]+padding > img.shape[0]:
        padding = 0

    cropped_img = img[(rect[1]-padding):(rect[1]+rect[3]+padding), (rect[0]-padding):(rect[0]+rect[2]+padding)]

    if cropped_img.shape[1] > 800:
        cropped_img = cv2.resize(cropped_img, dsize=(math.floor(cropped_img.shape[0]/(cropped_img.shape[1] / 800)), 800), interpolation=cv2.INTER_CUBIC)
    if cropped_img.shape[0] > 800:
        cropped_img = cv2.resize(cropped_img, dsize=(800, math.floor(cropped_img.shape[1]/(cropped_img.shape[0] / 800))), interpolation=cv2.INTER_CUBIC)
    #print(cropped_img.shape[0], cropped_img.shape[1])
    return cropped_img

def getFeatures(image, limits):
    no_bins = 16
    color_features_list = []
    magnitude_features_list = []
    orientation_features_list = []

    for x in range(0, image.shape[0], 8):
        for y in range(0, image.shape[1], 8):

            patch = image[x:x+16, y:y+16]
            color_histogram = getSpHistograms(patch, no_bins, limits)
            magnitude_histogram, orientation_histogram  = getTextureHistograms(patch, no_bins, False, limits)

            #patch_features = np.vstack((color_histogram, magnitude_histogram, orientation_histogram))
            color_features_list.append(color_histogram)
            magnitude_features_list.append(magnitude_histogram)
            orientation_features_list.append(orientation_histogram)

    color_features = np.column_stack(color_features_list)
    magnitude_features = np.column_stack(magnitude_features_list)
    orientation_features = np.column_stack(orientation_features_list)

    return np.asfortranarray(color_features), np.asfortranarray(magnitude_features), np.asfortranarray(orientation_features)

def getSpHistograms(patch, no_bins, limits):

    features = np.zeros(shape = [no_bins * 3, 1])

    try:
        binned_patch_r = np.floor(((patch[:,:,0] - limits.cr_min) * no_bins) / (limits.cr_max - limits.cr_min))
    except:
        binned_patch_r = patch[:,:,0]

    try:
        binned_patch_g = np.floor(((patch[:,:,1] - limits.cg_min) * no_bins) / (limits.cg_max - limits.cg_min))
    except:
        binned_patch_g = patch[:,:,1]

    try:
        binned_patch_b = np.floor(((patch[:,:,2] - limits.cb_min) * no_bins) / (limits.cb_max - limits.cb_min))
    except:
        binned_patch_b = patch[:,:,2]
    #print(np.amin(binned_patch), np.amax(binned_patch))

    for i in range(0, no_bins):
        features[i] = (binned_patch_r[:] == i).sum()
        features[i + no_bins] = (binned_patch_g[:] == i).sum()
        features[i + no_bins * 2] = (binned_patch_b[:] == i).sum()

    features[15] += np.count_nonzero(binned_patch_r[:] == 16)
    features[31] += np.count_nonzero(binned_patch_g[:] == 16)
    features[47] += np.count_nonzero(binned_patch_b[:] == 16)

    features = features / (patch.shape[0] * patch.shape[1])
    #print(np.sum(features, axis=0))
    #print('color')
    #print(features)
    return features

def getTextureHistograms(patch, no_bins, check_limits, limits):

    gx = [
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ]

    gy = [
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
    ]

    g_max = limits.g_max
    g_min = limits.g_min
    o_max = limits.o_max
    o_min = limits.o_min

    image_shape = patch.shape
    width = image_shape[1]
    height = image_shape[0]

    magnitude_patch = np.zeros(shape = [height, width])
    orientation_patch = np.zeros(shape = [height, width])

    color_channel = 2

    #print('Original patch')
    #print(patch.shape)
    #print(patch[:,:,color_channel])
    for y in range(0, height):

        for x in range(0, width):

            gradient_y = (
                gy[0][0] * get_pixel_safe(patch, x - 1, y - 1, color_channel) +
                gy[0][1] * get_pixel_safe(patch, x, y - 1, color_channel) +
                gy[0][2] * get_pixel_safe(patch, x + 1, y - 1, color_channel) +
                gy[2][0] * get_pixel_safe(patch, x - 1, y + 1, color_channel) +
                gy[2][1] * get_pixel_safe(patch, x, y + 1, color_channel) +
                gy[2][2] * get_pixel_safe(patch, x + 1, y + 1, color_channel)
            )

            gradient_x = (
                gx[0][0] * get_pixel_safe(patch, x - 1, y - 1, color_channel) +
                gx[0][2] * get_pixel_safe(patch, x + 1, y - 1, color_channel) +
                gx[1][0] * get_pixel_safe(patch, x - 1, y, color_channel) +
                gx[1][2] * get_pixel_safe(patch, x + 1, y, color_channel) +
                gx[2][0] * get_pixel_safe(patch, x - 1, y - 1, color_channel) +
                gx[2][2] * get_pixel_safe(patch, x + 1, y + 1, color_channel)
            )

            #print('Gradient x and y')
            #print(gradient_x, gradient_y)
            #print "Gradient X: " + str(gradient_x) + " Gradient Y: " + str(gradient_y)
            magnitude = math.sqrt(pow(gradient_x, 2) + pow(gradient_y, 2))


            if gradient_x != 0 and gradient_y != 0:
                orientation = math.atan2(gradient_y, gradient_x)
            else:
                orientation = 0

            magnitude_patch[y, x] = magnitude
            orientation_patch[y, x] = orientation

    if check_limits:

        magnitude_max = np.amax(magnitude_patch)
        magnitude_min = np.amin(magnitude_patch)
        orientation_max = np.amax(orientation_patch)
        orientation_min = np.amin(orientation_patch)

        return magnitude_max, magnitude_min, orientation_max, orientation_min
    print('Magnitude of gradient')
    magnitude_patch = getGradientHistogram(magnitude_patch, no_bins, g_max, g_min)
    print('Orientation of gradient')
    orientation_patch = getGradientHistogram(orientation_patch, no_bins, o_max, o_min)

    return magnitude_patch, orientation_patch

def get_pixel_safe(image, x, y, layer):

    try:
        return image[y, x, layer]
    except:
        return 0

def getGradientHistogram(patch, no_bins, max, min):

    features = np.zeros(shape = [no_bins, 1])
    #print('Raw magnitude and orientation')
    #print(patch)
    #print('Magnitude/Orientation patch')
    #print(patch)
    print(patch)
    if np.amax(patch) != 0:
        binned_patch = np.floor(((patch + np.absolute(min)) * 16) / (max + np.absolute(min)))
    else:
        binned_patch = patch
    #print(np.amax(np.floor(binned_patch)), np.amin(np.floor(binned_patch)))
    print('Normalized to 0-16')
    print(binned_patch)
    for i in range(0, no_bins):
        features[i] = (binned_patch[:] == i).sum()

    features[15] += np.count_nonzero(binned_patch == 16)
    #print(np.sum(features, axis=0))

    features = features / (patch.shape[0] * patch.shape[1])
    #print(np.sum(features, axis=0))
    #print('Texture')
    #print(features)
    return features

def createGrid(image_size, patch_size):
    grid = np.zeros(shape = image_size)
    number = 1
    for i in range(0, image_size[0], patch_size):
        for j in range(0, image_size[1], patch_size):
            grid[i : i + patch_size, j : j + patch_size] = number
            number += 1
    return grid

def color_norm(I):

    I = I / 255
    I_norm = np.zeros(shape = I.shape)

    s_R = np.power(sum(sum(np.power(I[:,:,0], 6))/(I.shape[0]*I.shape[1])),1/6)
    s_G = np.power(sum(sum(np.power(I[:,:,1], 6))/(I.shape[0]*I.shape[1])),1/6)
    s_B = np.power(sum(sum(np.power(I[:,:,2], 6))/(I.shape[0]*I.shape[1])),1/6)

    som = np.sqrt(np.power(s_R, 2) + np.power(s_G, 2) + np.power(s_B, 2))

    s_R = s_R/som
    s_G = s_G/som
    s_B = s_B/som

    I_norm[:,:,0] = I[:,:,0]/(s_R*np.sqrt(3))
    I_norm[:,:,1] = I[:,:,1]/(s_G*np.sqrt(3))
    I_norm[:,:,2] = I[:,:,2]/(s_B*np.sqrt(3))

    return I_norm

def replicateSamples(dataset, replicas):

    used_indexes = []

    for sample in dataset:
        if sample.label == 1:
            used_indexes.append(sample.id)

    for replica in replicas:
        if replica.id in used_indexes:
            dataset.append(replica)

    labels = labelsToArray(dataset)

    return dataset, labels

def concatenateFeatures(dataset):

    color_features_dataset = []
    magnitude_features_dataset = []
    #orientation_features_dataset = []

    for image in dataset:

        color_features_dataset.append(image.color_features)
        magnitude_features_dataset.append(image.magnitude_features)
        #orientation_features_dataset.append(image.orientation_features)

    color_features = np.column_stack(color_features_dataset)
    magnitude_features = np.column_stack(magnitude_features_dataset)
    #orientation_features = np.column_stack(orientation_features_dataset)

    return np.asfortranarray(color_features), np.asfortranarray(magnitude_features)

def calcError(errorMatrix):
    error = 0
    errorM = errorMatrix**2
    for i in range(0, len(errorM[:,1])):
        for j in range(0, len(errorM[1,:])):
            error = error + errorM[i,j]
    #error = error/np.size(errorMatrix)
    return error

def calculateError(dataset, D, type_of_features):
    set_error = np.empty(shape = [len(dataset), 1])
    set_std = np.empty(shape = [len(dataset), 1])
    args = {'lambda1' : 0.1, 'L' : 10, 'pos': True}
    i = 0
    for image in dataset:

        if type_of_features == 'color':
            features = np.asfortranarray(image.color_features)
        elif type_of_features == 'magnitude':
            features = np.asfortranarray(image.magnitude_features)
        else:
            features = np.asfortranarray(image.orientation_features)

        alpha0 = spams.lasso(features, D=D, **args)

        error_matrix = features - D*alpha0

        set_error[i] = calcError(error_matrix)
        set_std[i] = np.std(error_matrix)
        i += 1

    mean_error = np.mean(set_error)
    mean_std = np.mean(set_std)

    print('Mean error :', mean_error, ' ;Mean std: ', mean_std)

        #set_error tem o erro por imagem, falta fazer a média


def getHistograms(dataset, atoms, D, type_of_features):

    args = {'lambda1' : 0.1, 'L' : 10, 'pos': True}
    i = 0
    histograms = []
    for image in dataset:

        if type_of_features == 'color':
            features = np.asfortranarray(image.color_features)
        elif type_of_features == 'magnitude':
            features = np.asfortranarray(image.magnitude_features)
        else:
            features = np.asfortranarray(image.orientation_features)
        alpha0 = spams.lasso(features, D=D, **args).T

        array = alpha0.toarray()
        #abs = np.absolute(array)
        array = np.mean(array, axis = 0)
        array = array.reshape(1, atoms)
        histograms.append(array)

    total_features = np.concatenate( histograms, axis=0 )
    return total_features

def getDiscriminativeHistograms(dataset, atoms, D, type_of_features):

    D = np.split(D, 2, axis = 1)
    D_m = D[0]
    D_n = D[1]

    args = {'lambda1' : 0.1, 'L' : 10, 'pos': True}
    i = 0
    histograms = []
    for image in dataset:

        if type_of_features == 'color':
            features = np.asfortranarray(image.color_features)
        elif type_of_features == 'magnitude':
            features = np.asfortranarray(image.magnitude_features)
        else:
            features = np.asfortranarray(image.orientation_features)
        alpha_m = spams.lasso(features, D=D_m, **args).T
        alpha_n = spams.lasso(features, D=D_n, **args).T

        alpha_m = alpha_m.toarray()
        alpha_n = alpha_n.toarray()
        array = np.column_stack((alpha_m, alpha_n))
        #abs = np.absolute(array)
        array = np.mean(array, axis = 0)
        array = array.reshape(1, atoms)
        histograms.append(array)

    total_features = np.concatenate( histograms, axis=0 )
    return total_features

def DictionarySingleLinkageClustering(X, method, metric, treshold):

    #X = pdist(D, metric)
    D = []
    Z = linkage(X, method, metric)
    i = distance = 0
    idxs = []
    print(Z)

    treshold = np.max(Z[:,2])*treshold
    print(np.mean(Z[:,2]))
    atoms = len(X)
    while True:
        distance = Z[i,2]
        if distance > treshold:
            break
        cluster_1_idx = int(Z[i,0])
        cluster_2_idx = int(Z[i,1])
        #print(cluster_1_idx, cluster_2_idx)
        if cluster_1_idx < atoms:
            if cluster_2_idx < atoms:
                new_atom = (X[cluster_1_idx] + X[cluster_2_idx]) / 2
                D.append(new_atom)

                idxs.append(cluster_1_idx)
                idxs.append(cluster_2_idx)

            else:
                #print(cluster_1_idx, cluster_2_idx - 512, len(D))
                new_atom = (X[cluster_1_idx] + D[cluster_2_idx - atoms]) / 2
                D.append(new_atom)

                idxs.append(cluster_1_idx)

        else:
            if cluster_2_idx < atoms:
                new_atom = (D[cluster_1_idx - atoms] + X[cluster_2_idx]) / 2
                D.append(new_atom)

                idxs.append(cluster_2_idx)
            else:
                new_atom = (D[cluster_1_idx - atoms] + D[cluster_2_idx - atoms]) / 2
                D.append(new_atom)

        i += 1
    idxs.sort(reverse = True)
    idxs_to_keep = list(range(atoms))
    for idx in idxs:
        idxs_to_keep.pop(idx)
    X = X[idxs_to_keep, :]
    D = np.concatenate(D)
    print(X.shape)
    print(D.shape)

    return X, D

def labelsToArray(data):
    labels = np.zeros(shape = [len(data), 1])

    for i in range(0, len(data)):
        labels[i] = data[i].label
    return labels

def getClassifier(training_data, labels, classifier_type, c, g, w):

    if classifier_type == 'SVM':

        weights = np.ones(labels.shape) + w*labels
        weights = np.ravel(weights)
        #class_weight="balanced
        #sample_weight = weights
        classifier = svm.SVC(C=c, gamma=g, probability=True)
        classifier.fit(training_data, labels)
        #print('C = %.3f and gamma = %.3f' % (c, g))
    elif classifier_type == 'RandomForest':
        classifier = RandomForestClassifier(n_estimators=c, max_depth=g, random_state=0)
        classifier.fit(training_data, labels)
    elif classifier_type == 'KNN':

        classifier = KNeighborsClassifier(n_neighbors=k)
        classifier.fit(training_data, labels)
        print('Done training KNN with ', k,' neighbours')

    elif classifier_type == 'KNNR':
        classifier = RadiusNeighborsClassifier(radius=1.0)
        classifier.fit(training_data, labels)
    else:
        print('Wrong classifier key name')


    return classifier

def predict(color_classifier, magnitude_classifier, orientation_classifier, color_histograms, magnitude_histograms, orientation_histograms, labels):

    color_prediction = color_classifier.predict_proba(color_histograms)
    magnitude_prediction = magnitude_classifier.predict_proba(magnitude_histograms)
    orientation_prediction = orientation_classifier.predict_proba(orientation_histograms)

    probabilities = (color_prediction + magnitude_prediction + orientation_prediction) / 3
    guess = 0
    #scores = color_classifier.decision_function(color_histograms)


    sensitivity = 0
    specitivity = 0
    se_color = sp_color = se_magnitude = sp_magnitude = se_orientation = sp_orientation = 0
    melanomas = np.count_nonzero(labels == 1)
    non_melanomas = len(labels) - melanomas

    for i in range(0, len(probabilities)):

        #print('Label: ', labels[i])
        #print('Color: ', color_prediction[i], 'Magnitude: ', magnitude_prediction[i], 'Orientation: ', orientation_prediction[i])

        if probabilities[i,0] > probabilities[i,1]:
            guess = 0
        else:
            guess = 1

        if guess == labels[i]:
            if labels[i] == 0:
                specitivity += 1
            else:
                sensitivity += 1


    cost = (1.5*(1 - (sensitivity/melanomas)) + 1*(1 - (specitivity/non_melanomas))) / (2.5)
    sensitivity = (sensitivity/melanomas) * 100
    specitivity = (specitivity/non_melanomas) * 100
    accuracy = (sensitivity + specitivity) / 2

    print('Accuracy = %.3f; Sensitivity = %.3f; Specitivity = %.3f' % (accuracy, sensitivity, specitivity))

    #print(labels.T
    #print(predictions)
    #se_color = sp_color = se_magnitude = sp_magnitude = se_orientation = sp_orientation = 0
    result = Result(sensitivity, specitivity, accuracy)
    return result

def predict2Classifiers(color_classifier, magnitude_classifier, color_histograms, magnitude_histograms, labels):

    color_probs = color_classifier.predict_proba(color_histograms)
    magnitude_probs = magnitude_classifier.predict_proba(magnitude_histograms)

    probabilities = (color_probs + magnitude_probs) / 2
    #probabilities = np.multiply(color_probs, magnitude_probs)
    guess = 0


    sensitivity = 0
    specitivity = 0
    se_color = sp_color = se_magnitude = sp_magnitude = se_orientation = sp_orientation = 0
    melanomas = np.count_nonzero(labels == 1)
    non_melanomas = len(labels) - melanomas

    for i in range(0, len(probabilities)):

        if probabilities[i,0] > probabilities[i,1]:
            guess = 0
        else:
            guess = 1

        if guess == labels[i]:
            if labels[i] == 0:
                specitivity += 1
            else:
                sensitivity += 1

    sensitivity = (sensitivity/melanomas) * 100
    specitivity = (specitivity/non_melanomas) * 100
    accuracy = (sensitivity + specitivity) / 2

    print('Accuracy = %.3f; Sensitivity = %.3f; Specitivity = %.3f' % (accuracy, sensitivity, specitivity))

    se_color = sp_color = se_magnitude = sp_magnitude = se_orientation = sp_orientation = 0
    result = Result(sensitivity, specitivity, accuracy)
    return result, color_probs, magnitude_probs

def predict2ClassifierswithProbs(color_probs, magnitude_probs, labels):

    c_result = getScoresFromPredictions(color_probs, labels)
    m_result = getScoresFromPredictions(magnitude_probs, labels)

    probabilities = (color_probs + magnitude_probs) / 2

    #fpr, tpr, tresholds = roc_curve(labels, probabilities[:, 1])

    #plt.figure(1)
    #plt.plot([0, 1], [0, 1], 'k--')
    #plt.plot(fpr, tpr, label='RT + LR')
    #plt.xlabel('False positive rate')
    #plt.ylabel('True positive rate')
    #plt.title('ROC curve')
    #plt.legend(loc='best')
    #plt.show()
    #probabilities = np.multiply(color_probs, magnitude_probs)
    result = getScoresFromPredictions(probabilities, labels)
    return result, c_result, m_result

def getScoresFromPredictions(probabilities, labels):

    predictions = np.zeros(shape = [len(labels)])

    guess = 0
    sensitivity = 0
    specitivity = 0
    se_color = sp_color = se_magnitude = sp_magnitude = se_orientation = sp_orientation = 0
    melanomas = np.count_nonzero(labels == 1)
    non_melanomas = len(labels) - melanomas

    for i in range(0, len(probabilities)):

        if probabilities[i,0] > probabilities[i,1]:
            guess = 0
        else:
            guess = 1
            predictions[i] = 1

        if guess == labels[i]:
            if labels[i] == 0:
                specitivity += 1
            else:
                sensitivity += 1

    sensitivity = (sensitivity/melanomas) * 100
    specitivity = (specitivity/non_melanomas) * 100
    accuracy = (sensitivity + specitivity) / 2

    print('Accuracy = %.3f; Sensitivity = %.3f; Specitivity = %.3f' % (accuracy, sensitivity, specitivity))

    result = Result(sensitivity, specitivity, accuracy)

    return result

def predictSingleClf(clf, features, labels):

    probabilities = clf.predict_proba(features)
    guess = 0
    sensitivity = 0
    specitivity = 0

    melanomas = np.count_nonzero(labels == 1)
    non_melanomas = len(labels) - melanomas

    for i in range(0, len(probabilities)):

        if probabilities[i,0] > probabilities[i,1]:
            guess = 0
        else:
            guess = 1

        if guess == labels[i]:
            if labels[i] == 0:
                specitivity += 1
            else:
                sensitivity += 1

    cost = (1.5*(1 - (sensitivity/melanomas)) + 1*(1 - (specitivity/non_melanomas))) / (2.5)

    sensitivity = (sensitivity/melanomas) * 100
    specitivity = (specitivity/non_melanomas) * 100
    accuracy = (sensitivity + specitivity) / 2

    print('Accuracy = %.3f; Sensitivity = %.3f; Specitivity = %.3f' % (accuracy, sensitivity, specitivity))

    return accuracy, sensitivity, specitivity, cost

def plotFeatures(data, labels, atoms):
    labels = labels.ravel()
    no_melanomas = np.count_nonzero(labels)
    melanomas = np.empty(shape = [no_melanomas, atoms])
    non_melanomas = np.empty(shape = [len(labels) - no_melanomas, atoms])
    ncount = 0
    mcount = 0
    for i in range(0, len(labels)):
        if labels[i] == 0:
            non_melanomas[ncount] = data[i,:]
            ncount += 1
        else:
            melanomas[mcount] = data[i,:]
            mcount += 1

    data = (melanomas, non_melanomas)
    colors = ("red", "blue")
    classes = ("melanoma", "non melanoma")

    for i in range(0, atoms - 1):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        for count, item in enumerate(data):
            plt.scatter(item[:,i], item[:,i+1], c=colors[count], label=classes[count])
        plt.title('Feature %s with feature %s' %(i, i+1))
        plt.legend(loc=1)
        filename = './figures/Features_%s_%s' %(i, i+1)
        plt.savefig(filename, bbox_inches="tight")
        plt.close()
        sys.stdout.write("\r{0}".format(round((float(i)/(atoms-1))*100,2)))
        sys.stdout.flush()

def featureSelection(features, labels):
    revised_features = np.empty(shape = [len(features[:,1]), 0])
    count = 0
    feature_no = np.empty(shape = [1,0])
    for i in range(0, len(features[0,:])):
        corr = pearsonr(features[:,i], labels.ravel())
        if np.absolute(corr[0]) > 0:
            revised_features = np.column_stack((revised_features, features[:,i]))
            count += 1
            feature_no = np.column_stack((feature_no, i))
    print('Number of selected features: ', count)
    return revised_features, feature_no

def getBestParameters(tuning, max):

    tuning.sort(key=lambda tuning: tuning.accuracy, reverse=True)

    best_parameters = tuning[0:max]

    for param in best_parameters:
        print('Parameters')
        print('Atoms = ', param.atoms, '; C = ',param.c, '; gamma = ', param.g)
        print('Scores')
        print('SE = ', param.sensitivity, '; SP = ', param.specitivity, '; BACC = ', param.accuracy)
    return best_parameters

def getBestParametersCost(tuning, cost=False):

    if cost == True:
        min_cost = 1000

        for parameters in tuning:

            if parameters.cost < min_cost:

                atoms = parameters.atoms
                c = parameters.c
                g = parameters.g
                w = parameters.w
                min_cost = parameters.cost
    else:
        tuning.sort(key=lambda tuning: tuning.accuracy, reverse=True)
        parameters = tuning[0]
        atoms = parameters.atoms
        c = parameters.c
        g = parameters.g
        w = parameters.w
        max_accuracy = parameters.accuracy
        print('SE = ', parameters.sensitivity, '; SP = ', parameters.specitivity, '; BACC = ', parameters.accuracy)

    return atoms, c, g, w, parameters.accuracy, parameters.sensitivity, parameters.specitivity

def getDictionary(Dictionaries, type, atoms, nested_no, cross_no, info):

    for Dic in Dictionaries:
        if ( Dic.type == type and Dic.nested_no == nested_no and
             Dic.cross_no == cross_no and Dic.info == info and Dic.atoms == atoms):
           return Dic.D

    print('No such dictionary exists')
    exit(0)

def checkImages(data, predictions):

    i = 0

    for image in data:

        path = './Data/Images/' + image.name
        image_to_save = misc.imread(path)

        if image.label == predictions[i]:
            if image.label == 1: # True positive
                new_path = './Images/edra/TruePositives/' + image.name
            else: # True negative
                new_path = './Images/edra/TrueNegative/' + image.name
        else:
            if image.label == 1: # False positive
                new_path = './Images/edra/FalsePositives/' + image.name
            else: #False negative
                new_path = './Images/edra/FalseNegative/' + image.name

        i += 1
        misc.imsave(new_path, image_to_save)

    return

def main():
    Discriminative = True
    load_dictionaries = True
    Discriminative_sparse_codes = False
    #color_classifier = joblib.load('saves/test_color_classifier.sav')
    #magnitude_classifier = joblib.load('saves/test_magnitude_classifier.sav')
    #test_color_histograms = joblib.load('saves/test_color_histograms.sav')
    #test_magnitude_histograms = joblib.load('saves/test_magnitude_histograms.sav')
    #test_labels = joblib.load('saves/test_labels.sav')
    #result = predict2Classifiers(color_classifier,
    #                    magnitude_classifier,
    #                    test_color_histograms, test_magnitude_histograms, test_labels)
    #return
    f = open('saves/logs/edra_log_14.txt', 'w')
    ####################### LOAD TRAINING DATASET #########################
    #image_names = joblib.load('saves/edra_image_names.sav')
    #labels = joblib.load('saves/edra_labels.sav')
    #dataset, replicas = getData('Images', labels, True)
    #for image, label in zip(image_names, labels):
    #    print(label,';  ', image)
    #return
    #filename = 'saves/training_dataset_overlapp.sav'
    #joblib.dump(dataset, 'saves/datasets/edra_dataset.sav')
    #joblib.dump(replicas, 'saves/datasets/edra_replicas.sav')
    dataset = joblib.load('saves/datasets/edra_dataset.sav')
    replicas = joblib.load('saves/datasets/edra_replicas.sav')
    Dictionaries = joblib.load('saves/dictionaries_edra.sav')
    ####################### MERGE DATASETS ################################

    labels = labelsToArray(dataset)
    dataset_size = len(dataset)
    #print('dataset size: ', dataset_size)
    #print('no of melanomas: ', np.count_nonzero(train_labels == 1))
    #C = [2**2, 2**3, 2**4, 2**5, 2**6, 2**7, 2**8]
    C = [50, 100, 200]
    gamma = [5, 10, 20]
    #gamma = [2**1, 2**2, 2**3, 2**4, 2**5]
    weights = [1]
    Experiments = []
    bacc = sensitivity = specitivity = 0
    ####################### TUNING ################################
    if True:
        #C = [2**-4, 2**-3, 2**-2, 2**-1, 2**0, 2**1, 2**2, 2**3, 2**4]
        #gamma = [2**-4, 2**-3, 2**-2, 2**-1, 2**0, 2**1, 2**2, 2**3, 2**4]
        #K = [2, 3, 4, 5, 6 ,7 ,8 ,9]

        #file.write('\nC range :', C,'\ngamma = ', gamma)

        training_fold = []
        validation_fold = []
        train_fold = []
        test_fold = []

        loop_count = 0
        test_fold_fraction = int(np.floor((dataset_size / 10)))
        for t in range(0, 10):

            c_tuning_results = []
            m_tuning_results = []
            #o_tuning_results = []

            begining_index = t * test_fold_fraction
            end_index = t * test_fold_fraction + test_fold_fraction

            ####################### GET FOLDS ######################
            #print(begining_index, end_index)
            train_fold = dataset[:begining_index] + dataset[end_index:]
            #training_fold, train_fold_labels = replicateSamples(training_fold, train_replicas)
            train_labels_for_ratio = labelsToArray(train_fold)

            test_fold = dataset[begining_index:end_index]

            #train_fold_labels = np.vstack((train_labels[:begining_index], train_labels[end_index:]))
            test_labels = labels[begining_index:end_index]

            fold_fraction = int(np.floor((len(train_fold) / 5)))
            f.write('Size of train fold %d \n' %(len(train_fold)))
            f.write('Percentage of melanomas in train fold: %.2f \n' %(np.count_nonzero(train_labels_for_ratio == 1)/len(train_fold)))
            f.write('Size of test fold %d \n' %(len(test_fold)))
            f.write('Percentage of melanomas in test fold: %.2f \n' %(np.count_nonzero(test_labels == 1)/len(test_fold)))
            for k in range(0, 5):

                print('Starting fold ', k)

                begining_index = k * fold_fraction
                end_index = k * fold_fraction + fold_fraction

                ####################### GET FOLDS ######################
                #print(begining_index, end_index)
                training_fold = train_fold[:begining_index] + train_fold[end_index:]
                training_fold, train_fold_labels = replicateSamples(training_fold, replicas)

                validation_fold = train_fold[begining_index:end_index]

                #train_fold_labels = np.vstack((train_labels[:begining_index], train_labels[end_index:]))
                validation_fold_labels = labelsToArray(validation_fold)

                if not(load_dictionaries):
                    if Discriminative:
                        ####################### SEPARATE CLASSES ######################

                        melanomas, non_melanomas = getClassSamples(training_fold)

                        m_color_features, m_magnitude_features = concatenateFeatures(melanomas)
                        n_color_features, n_magnitude_features = concatenateFeatures(non_melanomas)
                    else:
                        ####################### GET TRAINING FEATURES ######################

                        color_features, magnitude_features = concatenateFeatures(training_fold)

                #joblib.dump(color_features, 'saves/color_features.sav')
                #joblib.dump(magnitude_features, 'saves/magnitude_features.sav')
                #joblib.dump(orientation_features, 'saves/orientation_features.sav')
                #train_features = joblib.load('saves/training_features_upgrade_8.sav')
                print('Done loading training data')
                #f.write('Color parameter tuning for fold %d \n' %(k))
                for power_c in range(7, 9):
                    atoms = (2**power_c) * 2
                    #f.write('number of atoms %d \n' %(2**power_c))
                    if load_dictionaries:
                        D_color = getDictionary(Dictionaries, 'color', atoms, t, k, 'cross')
                    else:
                        param_c = { 'K' : (2**power_c),
                                    'lambda1' : 0.1 , 'lambda2' : 0 , 'numThreads' : 4, 'batchsize' : 100,
                                    'iter' : 500, 'posAlpha': True, 'verbose': False}
                        if Discriminative:
                            D_m_color = spams.trainDL(m_color_features ,**param_c)
                            D_n_color = spams.trainDL(n_color_features ,**param_c)
                            D_color = np.column_stack((D_m_color, D_n_color))

                              #melanomas = non_melanomas = []

                              #m_color_features = m_magnitude_features = m_orientation_features = []
                              #n_color_features = n_magnitude_features = n_orientation_features = []
                        else:
                            D_color = spams.trainDL(color_features ,**param_c)
                            atoms = 2**power_c

                    #Dic = Dictionary(D_color, atoms, 'color', t, k, 'cross')
                    #Dictionaries.append(Dic)

                    if Discriminative_sparse_codes:
                        train_histograms = getDiscriminativeHistograms(training_fold, atoms, D_color, 'color')
                        eval_histograms = getDiscriminativeHistograms(validation_fold, atoms, D_color, 'color')
                    else:
                        train_histograms = getHistograms(training_fold, atoms, D_color, 'color')
                        eval_histograms = getHistograms(validation_fold, atoms, D_color, 'color')

                    for c in C:
                        for g in gamma:
                            for w in weights:

                                classifier = getClassifier(train_histograms, np.ravel(train_fold_labels), 'RandomForest', c, g, w)
                                t_bacc, t_sensitivity, t_specitivity, cost = predictSingleClf(classifier, eval_histograms, np.ravel(validation_fold_labels))
                                #f.write('C = %d; g = %d; bacc = %.3f; se = %.3f; sp = %.3f \n' %(c, g, bacc, sensitivity, specitivity))
                                if k == 0:
                                    performance = Tuning(2**power_c, c, g, w, t_bacc, t_sensitivity, t_specitivity, cost)
                                    c_tuning_results.append(performance)
                                else:
                                    c_tuning_results[incrementor].accuracy += t_bacc
                                    c_tuning_results[incrementor].sensitivity += t_sensitivity
                                    c_tuning_results[incrementor].specitivity += t_specitivity
                                      #tuning_results[incrementor].result.se_color += result.se_color
                                      #tuning_results[incrementor].result.sp_color += result.sp_color
                                      #tuning_results[incrementor].result.se_magnitude += result.se_magnitude
                                      #tuning_results[incrementor].result.sp_magnitude += result.sp_magnitude
                                      #tuning_results[incrementor].result.se_orientation += result.se_orientation
                                      #tuning_results[incrementor].result.sp_orientation += result.sp_orientation
                                    incrementor += 1


                                sys.stdout.write("\r{0}".format(round((float(loop_count)/(10*5*4*(len(C)*len(gamma)*len(weights)*2)))*100,2)))
                                sys.stdout.flush()
                                loop_count += 1
                incrementor = 0
                #f.write('Magnitude parameter tuning for fold %d \n' %(k))
                for power_m in range(7, 9):
                    atoms = (2**power_m) * 2
                    if load_dictionaries:
                        D_magnitude = getDictionary(Dictionaries, 'magnitude', atoms, t, k, 'cross')
                    else:
                        param_m = { 'K' : (2**power_m),
                                  'lambda1' : 0.1 , 'lambda2' : 0 , 'numThreads' : 4, 'batchsize' : 100,
                                  'iter' : 500, 'posAlpha': True, 'verbose': False}
                        if Discriminative:
                            D_m_magnitude = spams.trainDL(m_magnitude_features ,**param_m)
                            D_n_magnitude = spams.trainDL(n_magnitude_features ,**param_m)
                            D_magnitude = np.column_stack((D_m_magnitude, D_n_magnitude))
                            #melanomas = non_melanomas = []

                            #m_color_features = m_magnitude_features = m_orientation_features = []
                            #n_color_features = n_magnitude_features = n_orientation_features = []
                        else:
                            D_magnitude = spams.trainDL(magnitude_features ,**param_m)
                            atoms = 2**power_m

                    #Dic = Dictionary(D_magnitude, atoms, 'magnitude', t, k, 'cross')
                    #Dictionaries.append(Dic)

                    if Discriminative_sparse_codes:
                        train_histograms = getDiscriminativeHistograms(training_fold, atoms, D_magnitude, 'magnitude')
                        eval_histograms = getDiscriminativeHistograms(validation_fold, atoms, D_magnitude, 'magnitude')
                    else:
                        train_histograms = getHistograms(training_fold, atoms, D_magnitude, 'magnitude')
                        eval_histograms = getHistograms(validation_fold, atoms, D_magnitude, 'magnitude')

                    print('Magnitude scores')
                    for c in C:
                        for g in gamma:
                            for w in weights:
                                classifier = getClassifier(train_histograms, np.ravel(train_fold_labels), 'SVM', c, g, w)
                                t_bacc, t_sensitivity, t_specitivity, cost = predictSingleClf(classifier, eval_histograms, np.ravel(validation_fold_labels))
                                #f.write('C = %d; g = %d; bacc = %.3f; se = %.3f; sp = %.3f \n' %(c, g, bacc, sensitivity, specitivity))
                                if k == 0:
                                    performance = Tuning(2**power_m, c, g, w, t_bacc, t_sensitivity, t_specitivity, cost)
                                    m_tuning_results.append(performance)
                                else:
                                    m_tuning_results[incrementor].accuracy += t_bacc
                                    m_tuning_results[incrementor].sensitivity += t_sensitivity
                                    m_tuning_results[incrementor].specitivity += t_specitivity
                                    #tuning_results[incrementor].result.se_color += result.se_color
                                    #tuning_results[incrementor].result.sp_color += result.sp_color
                                    #tuning_results[incrementor].result.se_magnitude += result.se_magnitude
                                    #tuning_results[incrementor].result.sp_magnitude += result.sp_magnitude
                                    #tuning_results[incrementor].result.se_orientation += result.se_orientation
                                    #tuning_results[incrementor].result.sp_orientation += result.sp_orientation
                                    incrementor += 1


                                sys.stdout.write("\r{0}".format(round((float(loop_count)/(10*5*4*(len(C)*len(gamma)*len(weights)*2)))*100,2)))
                                sys.stdout.flush()
                                loop_count += 1
                incrementor = 0

                ####################### PARAMETERS RESULTS UPDATE ############################

                #joblib.dump(c_tuning_results, 'saves/tuning_results/5_color_tuning_results.sav')
                #joblib.dump(m_tuning_results, 'saves/tuning_results/5_magnitude_tuning_results.sav')
                #joblib.dump(o_tuning_results, 'saves/tuning_results/5_tuning_results_orientation.sav')

            for i in range(0, len(c_tuning_results)):
                c_tuning_results[i].accuracy = c_tuning_results[i].accuracy / 5
                c_tuning_results[i].sensitivity = c_tuning_results[i].sensitivity / 5
                c_tuning_results[i].specitivity = c_tuning_results[i].specitivity / 5
                m_tuning_results[i].accuracy = m_tuning_results[i].accuracy / 5
                m_tuning_results[i].sensitivity = m_tuning_results[i].sensitivity / 5
                m_tuning_results[i].specitivity = m_tuning_results[i].specitivity / 5
                #o_tuning_results[i].accuracy = o_tuning_results[i].accuracy / 5
                #o_tuning_results[i].sensitivity = o_tuning_results[i].sensitivity / 5
                #o_tuning_results[i].specitivity = o_tuning_results[i].specitivity / 5

            #joblib.dump(c_tuning_results, 'saves/tuning_results/5_color_tuning_results.sav')
            #joblib.dump(m_tuning_results, 'saves/tuning_results/5_magnitude_tuning_results.sav')


            #joblib.dump(o_tuning_results, 'saves/tuning_results/5_tuning_results_orientation.sav')
            ####################### GET BEST PARAMETERS ############################
            f.write('\n \n ############## Testing ############ \n \n')
            print('Color')
            atoms_c, c_c, c_g, c_w, t_bacc, t_se, t_sp = getBestParametersCost(c_tuning_results)
            f.write('Color best parameters \n')
            f.write('Atoms = %d; C = %d; g = %d; bacc = %.3f; se = %.3f; sp = %.3f \n' %(atoms_c, c_c, c_g, t_bacc, t_se, t_sp))
            print('Magnitude')
            atoms_m, m_c, m_g, m_w, t_bacc, t_se, t_sp = getBestParametersCost(m_tuning_results)
            f.write('Magnitude best parameters \n')
            f.write('Atoms = %d; C = %d; g = %d; bacc = %.3f; se = %.3f; sp = %.3f \n' %(atoms_m, m_c, m_g, t_bacc, t_se, t_sp))
            #print('Orientation')
            #atoms_o, o_c, o_g, o_w, bacc, se, sp = getBestParametersCost(o_tuning_results)
            #f.write('Orientation best parameters \n')
            #f.write('Atoms = %d; C = %d; g = %d; bacc = %.3f; se = %.3f; sp = %.3f \n' %(atoms_o, o_c, o_g, bacc, se, sp))



            train_dataset, train_labels = replicateSamples(train_fold, replicas)
            shuffle(train_dataset)
            train_labels = labelsToArray(train_dataset)

            ####################### GET TRAINING FEATURES ######################

            if not(load_dictionaries):
                if Discriminative:

                    melanomas, non_melanomas = getClassSamples(train_dataset)

                    m_color_features, m_magnitude_features = concatenateFeatures(melanomas)
                    n_color_features, n_magnitude_features = concatenateFeatures(non_melanomas)
                else:

                    color_features, magnitude_features = concatenateFeatures(train_dataset)


            ####################### LEARN DICTIONARIES ############################


            if load_dictionaries:
                atoms_c = atoms_c * 2
                atoms_m = atoms_m * 2
                D_color = getDictionary(Dictionaries, 'color', atoms_c, t, k, 'nested')
                D_magnitude = getDictionary(Dictionaries, 'magnitude', atoms_m, t, k, 'nested')
            else:
                param_c = { 'K' : (atoms_c),
                          'lambda1' : 0.1 , 'lambda2' : 0 , 'numThreads' : 4, 'batchsize' : 100,
                          'iter' : 500, 'posAlpha': True, 'verbose': False}
                param_m = { 'K' : (atoms_m),
                          'lambda1' : 0.1 , 'lambda2' : 0 , 'numThreads' : 4, 'batchsize' : 100,
                          'iter' : 500, 'posAlpha': True, 'verbose': False}
                #param_o = { 'K' : (atoms_o),
                #          'lambda1' : 0.1 , 'lambda2' : 0 , 'numThreads' : 4, 'batchsize' : 100,
                #          'iter' : 500, 'posAlpha': True}

                if Discriminative:
                    D_m_color = spams.trainDL(m_color_features ,**param_c)
                    D_n_color = spams.trainDL(n_color_features ,**param_c)
                    D_color = np.column_stack((D_m_color, D_n_color))

                    D_m_magnitude = spams.trainDL(m_magnitude_features ,**param_m)
                    D_n_magnitude = spams.trainDL(n_magnitude_features ,**param_m)
                    D_magnitude = np.column_stack((D_m_magnitude, D_n_magnitude))

                    #D_m_orientation = spams.trainDL(m_orientation_features ,**param_o)
                    #D_n_orientation = spams.trainDL(n_orientation_features ,**param_o)
                    #D_orientation = np.column_stack((D_m_orientation, D_n_orientation))
                    atoms_c = atoms_c * 2
                    atoms_m = atoms_m * 2
                    melanomas = non_melanomas = []

                    #atoms_o = atoms_o * 2
                    #m_color_features = m_magnitude_features = m_orientation_features = []
                    #n_color_features = n_magnitude_features = n_orientation_features = []
                else:
                    D_color = spams.trainDL(color_features ,**param_c)
                    D_magnitude = spams.trainDL(magnitude_features ,**param_m)
                    #D_orientation = spams.trainDL(orientation_features ,**param_o)

            ####################### GET TRAIN HISTOGRAMS ############################

            if Discriminative_sparse_codes:
                train_color_histograms = getDiscriminativeHistograms(train_dataset, atoms_c, D_color, 'color')
                train_magnitude_histograms = getDiscriminativeHistograms(train_dataset, atoms_m, D_magnitude, 'magnitude')
            else:
                train_color_histograms = getHistograms(train_dataset, atoms_c, D_color, 'color')
                train_magnitude_histograms = getHistograms(train_dataset, atoms_m, D_magnitude, 'magnitude')

            ####################### GET TEST HISTOGRAMS ############################

            if Discriminative_sparse_codes:
                test_color_histograms = getDiscriminativeHistograms(test_fold, atoms_c, D_color, 'color')
                test_magnitude_histograms = getDiscriminativeHistograms(test_fold, atoms_m, D_magnitude, 'magnitude')
            else:
                test_color_histograms = getHistograms(test_fold, atoms_c, D_color, 'color')
                test_magnitude_histograms = getHistograms(test_fold, atoms_m, D_magnitude, 'magnitude')

            ####################### TRAIN CLASSIFIERS ############################

            color_classifier = getClassifier(train_color_histograms, np.ravel(train_labels), 'RandomForest', c_c, c_g, c_w)
            magnitude_classifier = getClassifier(train_magnitude_histograms, np.ravel(train_labels), 'RandomForest', m_c, m_g, m_w)
            #orientation_classifier = getClassifier(train_orientation_histograms, np.ravel(train_labels), 'SVM', m_c, m_g, m_w)

            c_bacc, c_sensitivity, c_specitivity, cost = predictSingleClf(color_classifier, test_color_histograms, np.ravel(test_labels))
            m_bacc, m_sensitivity, m_specitivity, cost = predictSingleClf(magnitude_classifier, test_magnitude_histograms, np.ravel(test_labels))

            f.write('\n \n ############## Testing ############ \n \n')
            f.write('Individual Test results for test fold %d \n' %(t))
            f.write('\nColor\n')
            f.write('bacc = %.3f; se = %.3f; sp = %.3f \n' %(c_bacc, c_sensitivity, c_specitivity))
            f.write('\nMagnitude\n')
            f.write('bacc = %.3f; se = %.3f; sp = %.3f \n' %(m_bacc, m_sensitivity, m_specitivity))

            ####################### GET RESULTS ###################################
            #print('For color C=',c_c,'; color gamma=',c_g,'mag C=',m_c,';mag gamma=',m_g)
            result, c_predictions, m_predictions = predict2Classifiers(color_classifier, magnitude_classifier,
                                test_color_histograms, test_magnitude_histograms, test_labels)

            current_experiment = Experiment(c_tuning_results, m_tuning_results, c_predictions, m_predictions)
            Experiments.append(current_experiment)

            bacc = bacc + result.accuracy
            sensitivity = sensitivity + result.sensitivity
            specitivity = specitivity + result.specitivity

            f.write('Test results for test fold %d \n' %(t))
            f.write('bacc = %.3f; se = %.3f; sp = %.3f \n' %(result.accuracy, result.sensitivity, result.specitivity))
            f.write('\n \n ############## Testing ############ \n \n')
        print('Bacc = ', bacc/10 ,'%; SE = ', sensitivity/10,'%; SP = ', specitivity/10,'%')
        f.write('Final results\n')
        f.write('bacc = %.3f; se = %.3f; sp = %.3f \n' %(bacc/10, sensitivity/10, specitivity/10))
        f.close()
        joblib.dump(Experiments, 'saves/Experiments/edra_experiment_14.sav')
        #joblib.dump(Dictionaries, 'saves/dictionaries_edra.sav')
    elif False:
        Experiments = joblib.load('saves/Experiments/edra_experiment_3.sav')
        test_fold_fraction = int(np.floor((dataset_size / 10)))
        for t in range(10):

            begining_index = t * test_fold_fraction
            end_index = t * test_fold_fraction + test_fold_fraction

            ####################### GET FOLDS ######################
            #print(begining_index, end_index)
            train_fold = dataset[:begining_index] + dataset[end_index:]

            test_fold = dataset[begining_index:end_index]

            #train_fold_labels = np.vstack((train_labels[:begining_index], train_labels[end_index:]))
            test_labels = labels[begining_index:end_index]


            c_tuning_results = Experiments[t].c_tuning_results
            m_tuning_results = Experiments[t].m_tuning_results

            f.write('\n \n ############## Testing ############ \n \n')
            print('Color')
            atoms_c, c_c, c_g, c_w, bacc, se, sp = getBestParametersCost(c_tuning_results)
            f.write('Color best parameters \n')
            f.write('Atoms = %d; C = %d; g = %d; bacc = %.3f; se = %.3f; sp = %.3f \n' %(atoms_c, c_c, c_g, bacc, se, sp))
            print('Magnitude')
            atoms_m, m_c, m_g, m_w, bacc, se, sp = getBestParametersCost(m_tuning_results)
            f.write('Magnitude best parameters \n')
            f.write('Atoms = %d; C = %d; g = %d; bacc = %.3f; se = %.3f; sp = %.3f \n' %(atoms_m, m_c, m_g, bacc, se, sp))
            #print('Orientation')
            #atoms_o, o_c, o_g, o_w, bacc, se, sp = getBestParametersCost(o_tuning_results)
            #f.write('Orientation best parameters \n')
            #f.write('Atoms = %d; C = %d; g = %d; bacc = %.3f; se = %.3f; sp = %.3f \n' %(atoms_o, o_c, o_g, bacc, se, sp))


            train_dataset, train_labels = replicateSamples(train_fold, replicas)
            shuffle(train_dataset)
            train_labels = labelsToArray(train_dataset)

            ####################### GET TRAINING FEATURES ######################

            if Discriminative:

                melanomas, non_melanomas = getClassSamples(train_dataset)

                m_color_features, m_magnitude_features = concatenateFeatures(melanomas)
                n_color_features, n_magnitude_features = concatenateFeatures(non_melanomas)
            else:

                color_features, magnitude_features = concatenateFeatures(train_dataset)

            ####################### LEARN DICTIONARIES ############################


            param_c = { 'K' : (atoms_c),
                      'lambda1' : 0.1 , 'lambda2' : 0 , 'numThreads' : 4, 'batchsize' : 100,
                      'iter' : 500, 'posAlpha': True}
            param_m = { 'K' : (atoms_m),
                      'lambda1' : 0.1 , 'lambda2' : 0 , 'numThreads' : 4, 'batchsize' : 100,
                      'iter' : 500, 'posAlpha': True}
            #param_o = { 'K' : (atoms_o),
            #          'lambda1' : 0.1 , 'lambda2' : 0 , 'numThreads' : 4, 'batchsize' : 100,
            #          'iter' : 500, 'posAlpha': True}

            if Discriminative:
                D_m_color = spams.trainDL(m_color_features ,**param_c)
                D_n_color = spams.trainDL(n_color_features ,**param_c)
                #D_color = np.column_stack((D_m_color, D_n_color))

                D_m_magnitude = spams.trainDL(m_magnitude_features ,**param_m)
                D_n_magnitude = spams.trainDL(n_magnitude_features ,**param_m)
                #D_magnitude = np.column_stack((D_m_magnitude, D_n_magnitude))

                #D_m_orientation = spams.trainDL(m_orientation_features ,**param_o)
                #D_n_orientation = spams.trainDL(n_orientation_features ,**param_o)
                #D_orientation = np.column_stack((D_m_orientation, D_n_orientation))

                melanomas = non_melanomas = []
                atoms_c = atoms_c * 2
                atoms_m = atoms_m * 2
                #atoms_o = atoms_o * 2
                #m_color_features = m_magnitude_features = m_orientation_features = []
                #n_color_features = n_magnitude_features = n_orientation_features = []
            else:
                D_color = spams.trainDL(color_features ,**param_c)
                D_magnitude = spams.trainDL(magnitude_features ,**param_m)
                #D_orientation = spams.trainDL(orientation_features ,**param_o)

            ####################### GET TRAIN HISTOGRAMS ############################

            train_color_histograms = getDiscriminativeHistograms(train_dataset, atoms_c, D_m_color, D_n_color, 'color')
            train_magnitude_histograms = getDiscriminativeHistograms(train_dataset, atoms_m, D_m_magnitude, D_n_magnitude, 'magnitude')
            #train_orientation_histograms = getHistograms(train_dataset, atoms_o, D_orientation, 'orientation')

            ####################### GET TEST HISTOGRAMS ############################

            test_color_histograms = getDiscriminativeHistograms(test_fold, atoms_c, D_m_color, D_n_color, 'color')
            test_magnitude_histograms = getDiscriminativeHistograms(test_fold, atoms_m, D_m_magnitude, D_n_magnitude, 'magnitude')
            #test_orientation_histograms = getHistograms(test_fold, atoms_o, D_orientation, 'orientation')

            ####################### TRAIN CLASSIFIERS ############################

            color_classifier = getClassifier(train_color_histograms, np.ravel(train_labels), 'SVM', c_c, c_g, c_w)
            magnitude_classifier = getClassifier(train_magnitude_histograms, np.ravel(train_labels), 'SVM', m_c, m_g, m_w)
            #orientation_classifier = getClassifier(train_orientation_histograms, np.ravel(train_labels), 'SVM', m_c, m_g, m_w)

            c_bacc, c_sensitivity, c_specitivity, cost = predictSingleClf(color_classifier, test_color_histograms, np.ravel(test_labels))
            m_bacc, m_sensitivity, m_specitivity, cost = predictSingleClf(magnitude_classifier, test_magnitude_histograms, np.ravel(test_labels))

            f.write('\n \n ############## Testing ############ \n \n')
            f.write('Individual Test results for test fold %d \n' %(t))
            f.write('\nColor\n')
            f.write('bacc = %.3f; se = %.3f; sp = %.3f \n' %(c_bacc, c_sensitivity, c_specitivity))
            f.write('\nMagnitude\n')
            f.write('bacc = %.3f; se = %.3f; sp = %.3f \n' %(m_bacc, m_sensitivity, m_specitivity))

            ####################### GET RESULTS ###################################
            #print('For color C=',c_c,'; color gamma=',c_g,'mag C=',m_c,';mag gamma=',m_g)
            result, c_predictions, m_predictions = predict2Classifiers(color_classifier, magnitude_classifier,
                                test_color_histograms, test_magnitude_histograms, test_labels)

            bacc = bacc + result.accuracy
            sensitivity = sensitivity + result.sensitivity
            specitivity = specitivity + result.specitivity

            f.write('Test results for test fold %d \n' %(t))
            f.write('bacc = %.3f; se = %.3f; sp = %.3f \n' %(result.accuracy, result.sensitivity, result.specitivity))
            f.write('\n \n ############## Testing ############ \n \n')
        #print('Orientation')
        #for parameter in o_tuning_results:
        #    print(parameter.atoms, parameter.c, parameter.g, parameter.w, parameter.accuracy, parameter.sensitivity, parameter.specitivity)
        #    file.write('\nAtoms: %d; C = %0.4f; gamma = %0.4f; BACC = %0.2f ' % (parameter.atoms, parameter.c, parameter.g, parameter.accuracy))
        print('Bacc = ', bacc/10 ,'%; SE = ', sensitivity/10,'%; SP = ', specitivity/10,'%')
        f.write('Final results\n')
        f.write('bacc = %.3f; se = %.3f; sp = %.3f \n' %(bacc/10, sensitivity/10, specitivity/10))
        file.close()

    elif False:
        Experiments = joblib.load('saves/Experiments/edra_experiment_3.sav')
        f.write('\n \n ############## Testing ############ \n \n')
        test_fold_fraction = int(np.floor((dataset_size / 10)))
        final_result = Result(0,0,0)
        final_c_result = Result(0,0,0)
        final_m_result = Result(0,0,0)
        for t in range(10):

            begining_index = t * test_fold_fraction
            end_index = t * test_fold_fraction + test_fold_fraction

            test_labels = labels[begining_index:end_index]

            result, c_result, m_result = predict2ClassifierswithProbs(Experiments[t].c_predictions, Experiments[t].m_predictions, test_labels)

            c_tuning_results = Experiments[t].c_tuning_results
            m_tuning_results = Experiments[t].m_tuning_results

            f.write('\n Best parameters for fold %d \n' %(t))
            atoms_c, c_c, c_g, c_w, t_bacc, t_se, t_sp = getBestParametersCost(c_tuning_results)
            f.write('Color best parameters \n')
            f.write('Atoms = %d; C = %d; g = %d \n' %(atoms_c, c_c, c_g))
            print('Magnitude')
            atoms_m, m_c, m_g, m_w, t_bacc, t_se, t_sp = getBestParametersCost(m_tuning_results)
            f.write('Magnitude best parameters \n')
            f.write('Atoms = %d; C = %d; g = %d \n' %(atoms_m, m_c, m_g))

            f.write('\n \n ############## Testing ############ \n \n')
            f.write('Individual Test results for test fold %d \n' %(t))
            f.write('\nColor\n')
            f.write('bacc = %.3f; se = %.3f; sp = %.3f \n' %(c_result.accuracy, c_result.sensitivity, c_result.specitivity))
            f.write('\nMagnitude\n')
            f.write('bacc = %.3f; se = %.3f; sp = %.3f \n' %(m_result.accuracy, m_result.sensitivity, m_result.specitivity))
            f.write('Test results for test fold %d \n' %(t))
            f.write('bacc = %.3f; se = %.3f; sp = %.3f \n' %(result.accuracy, result.sensitivity, result.specitivity))

            final_result.add(result.accuracy, result.sensitivity, result.specitivity)
            final_c_result.add(c_result.accuracy, c_result.sensitivity, c_result.specitivity)
            final_m_result.add(m_result.accuracy, m_result.sensitivity, m_result.specitivity)

        f.write('\n \n ############## Final results ############ \n \n')
        f.write('\nColor\n')
        f.write('bacc = %.3f; se = %.3f; sp = %.3f \n' %(final_c_result.accuracy/10, final_c_result.sensitivity/10, final_c_result.specitivity/10))
        f.write('\nMagnitude\n')
        f.write('bacc = %.3f; se = %.3f; sp = %.3f \n' %(final_m_result.accuracy/10, final_m_result.sensitivity/10, final_m_result.specitivity/10))
        f.write('\n ############## Global Test results ############ \n')
        f.write('bacc = %.3f; se = %.3f; sp = %.3f \n' %(final_result.accuracy/10, final_result.sensitivity/10, final_result.specitivity/10))
        f.close()

        print(' ############## Final results ############')
        print('Color')
        print('bacc = %.3f; se = %.3f; sp = %.3f' %(final_c_result.accuracy/10, final_c_result.sensitivity/10, final_c_result.specitivity/10))
        print('Magnitude')
        print('bacc = %.3f; se = %.3f; sp = %.3f' %(final_m_result.accuracy/10, final_m_result.sensitivity/10, final_m_result.specitivity/10))
        print('############## Global Test results ############')
        print('bacc = %.3f; se = %.3f; sp = %.3f' %(final_result.accuracy/10, final_result.sensitivity/10, final_result.specitivity/10))


    else:
        c_tuning_results = joblib.load('saves/tuning_results/3_color_tuning_results.sav')
        m_tuning_results = joblib.load('saves/tuning_results/3_magnitude_tuning_results.sav')

    return

    #file.write('\nFinished cross-validation, Staring test')
    #print('Color')
    #for parameter in c_tuning_results:
    #    print(parameter.atoms, parameter.c, parameter.g, parameter.w, parameter.accuracy, parameter.sensitivity, parameter.specitivity)
    #print('Magnitude')
    #for parameter in m_tuning_results:
    #    print(parameter.atoms, parameter.c, parameter.g, parameter.w, parameter.accuracy, parameter.sensitivity, parameter.specitivity)
    #print('Orientation')
    #for parameter in o_tuning_results:
    #    print(parameter.atoms, parameter.c, parameter.g, parameter.w, parameter.accuracy, parameter.sensitivity, parameter.specitivity)

    atoms_c, c_c, c_g, c_w, bacc, se, sp = getBestParametersCost(c_tuning_results)
    atoms_m, m_c, m_g, m_w, bacc, se, sp = getBestParametersCost(m_tuning_results)
    #atoms_o, o_c, o_g, o_w = getBestParametersCost(o_tuning_results)
    atoms_c = 256
    atoms_m = 256
    #print('Color')
    #c_best_parameters = getBestParameters(c_tuning_results, 3)
    #print('Magnitude')
    #m_best_parameters = getBestParameters(m_tuning_results, 3)

    #print('Best Parameters are atoms_c = ', atoms_c, '; atoms_m = ', atoms_m,
    #        ';color C = ', c_c, ';color Gamma = ', c_g, ';color w = ', c_w,
    #        ';magnitude C = ', m_c, ';magnitude Gamma = ', m_g, ';magnitude w = ', m_w)

    ####################Let's see how the training did ####################

    train_accuracy = train_sensitivity = train_specitivity = 0
    k = inc = 0
    best_results = []
    if False:

        training_fold = []
        validation_fold = []

        tuning_results = []

        loop_count = 0
        fold_fraction = int(np.floor((dataset_size / 5)))
        for k in range(0, 5):
            inc = 0
            print('Starting fold ', k)

            begining_index = k * fold_fraction
            end_index = k * fold_fraction + fold_fraction

            ####################### GET FOLDS ######################

            training_fold = train_dataset[:begining_index] + train_dataset[end_index:]
            training_fold, train_fold_labels = replicateSamples(training_fold, train_replicas)

            validation_fold = train_dataset[begining_index:end_index]

            #train_fold_labels = np.vstack((train_labels[:begining_index], train_labels[end_index:]))
            validation_fold_labels = train_labels[begining_index:end_index]

            if Discriminative:
                ####################### SEPARATE CLASSES ######################

                melanomas, non_melanomas = getClassSamples(training_fold)

                m_color_features, m_magnitude_features = concatenateFeatures(melanomas)
                n_color_features, n_magnitude_features = concatenateFeatures(non_melanomas)
            else:
                ####################### GET TRAINING FEATURES ######################

                color_features, magnitude_features, orientation_features = concatenateFeatures(training_fold)

            print('Done loading training data')

            param_c = { 'K' : (atoms_c),
                      'lambda1' : 0.1 , 'lambda2' : 0 , 'numThreads' : 4, 'batchsize' : 100,
                      'iter' : 500, 'posAlpha': True}

            param_m = { 'K' : (atoms_m),
                      'lambda1' : 0.1 , 'lambda2' : 0 , 'numThreads' : 4, 'batchsize' : 100,
                      'iter' : 500, 'posAlpha': True}
            #param_o = { 'K' : (atoms_o),
            #          'lambda1' : 0.1 , 'lambda2' : 0 , 'numThreads' : 4, 'batchsize' : 100,
            #          'iter' : 500}
            #print('Using ', 2**power, ' Atoms')
            #file.write('\nUsing ', 2**power, ' Atoms')
            ####################### LEARN DICTIONARIES ############################

            if Discriminative:
                D_m_color = spams.trainDL(m_color_features ,**param_c)
                D_m_magnitude = spams.trainDL(m_magnitude_features ,**param_m)
            #    D_m_orientation = spams.trainDL(m_orientation_features ,**param_o)
                D_n_color = spams.trainDL(n_color_features ,**param_c)
                D_n_magnitude = spams.trainDL(n_magnitude_features ,**param_m)
            #    D_n_orientation = spams.trainDL(n_orientation_features ,**param_o)
                D_color = np.column_stack((D_m_color, D_n_color))
                D_magnitude = np.column_stack((D_m_magnitude, D_n_magnitude))
            #    D_orientation = np.column_stack((D_m_orientation, D_n_orientation))

            else:
                D_color = spams.trainDL(color_features ,**param_c)
                D_magnitude = spams.trainDL(magnitude_features ,**param_m)
            #    D_orientation = spams.trainDL(orientation_features ,**param_o)

            ####################### GET TRAIN HISTOGRAMS ############################

            train_color_histograms = getHistograms(training_fold, param_c['K']*2, D_color, 'color')
            train_magnitude_histograms = getHistograms(training_fold, param_m['K']*2, D_magnitude, 'magnitude')
            #train_orientation_histograms = getHistograms(training_fold, param_o['K'], D_orientation, 'orientation')

            print('Train histograms obtained')
            ####################### GET VALIDATION HISTOGRAMS ############################

            eval_color_histograms = getHistograms(validation_fold, param_c['K']*2, D_color, 'color')
            eval_magnitude_histograms = getHistograms(validation_fold, param_m['K']*2, D_magnitude, 'magnitude')
            #eval_orientation_histograms = getHistograms(validation_fold, param_o['K'], D_orientation, 'orientation')

            color_classifier = getClassifier(train_color_histograms, np.ravel(train_fold_labels), 'SVM', c_c, c_g, c_w)
            magnitude_classifier = getClassifier(train_magnitude_histograms, np.ravel(train_fold_labels), 'SVM', m_c, m_g, m_w)
            #orientation_classifier = getClassifier(train_orientation_histograms, np.ravel(train_fold_labels), 'SVM', o_c, o_g, o_w)

            ####################### GET RESULTS ###################################

            #result = predict(color_classifier,
            #                    magnitude_classifier, orientation_classifier,
            #                    eval_color_histograms, eval_magnitude_histograms,
            #                    eval_orientation_histograms, validation_fold_labels)

            result = predict2Classifiers(color_classifier,
                                magnitude_classifier,
                                eval_color_histograms, eval_magnitude_histograms, validation_fold_labels)

    ##################### Check parameters in testing set #########################

    train_dataset = dataset + replicas
    shuffle(train_dataset)
    train_labels = labelsToArray(train_dataset)

    ####################### GET TRAINING FEATURES ######################

    if Discriminative:

        melanomas, non_melanomas = getClassSamples(train_dataset)

        m_color_features, m_magnitude_features = concatenateFeatures(melanomas)
        n_color_features, n_magnitude_features = concatenateFeatures(non_melanomas)
    else:
        melanomas, non_melanomas = getClassSamples(train_dataset)
        color_features, magnitude_features = concatenateFeatures(train_dataset)

    ####################### LEARN DICTIONARIES ############################


    param_c = { 'K' : (atoms_c),
              'lambda1' : 0.1 , 'lambda2' : 0 , 'numThreads' : 4, 'batchsize' : 100,
              'iter' : 50, 'posAlpha': True}
    param_m = { 'K' : (atoms_m),
              'lambda1' : 0.1 , 'lambda2' : 0 , 'numThreads' : 4, 'batchsize' : 100,
              'iter' : 50, 'posAlpha': True}

    if Discriminative:
        D_m_color = spams.trainDL(m_color_features ,**param_c)
        D_n_color = spams.trainDL(n_color_features ,**param_c)
        D_color = np.column_stack((D_m_color, D_n_color))

        D_m_magnitude = spams.trainDL(m_magnitude_features ,**param_m)
        D_n_magnitude = spams.trainDL(n_magnitude_features ,**param_m)
        D_magnitude = np.column_stack((D_m_magnitude, D_n_magnitude))

        #melanomas = non_melanomas = []

        #m_color_features = m_magnitude_features = m_orientation_features = []
        #n_color_features = n_magnitude_features = n_orientation_features = []
    else:
        D_color = spams.trainDL(color_features ,**param_c)
        D_magnitude = spams.trainDL(magnitude_features ,**param_m)

    D, D_common = DictionarySingleLinkageClustering(D_color.T, 'single', 'correlation', 0.5)


    return
    ####################### GET TRAIN HISTOGRAMS ############################

    train_color_histograms = getDiscriminativeHistograms(train_dataset, atoms_c, D_m_color, D_n_color, 'color')

    print('Color')
    print('melanomas with non Discriminative dictionary')
    calculateError(melanomas, D_color, 'color')
    print('non melanomas with non Discriminative dictionary')
    calculateError(non_melanomas, D_color, 'color')

    print('magnitude')
    print('melanomas with non Discriminative dictionary')
    calculateError(melanomas, D_magnitude, 'magnitude')
    print('non melanomas with non Discriminative dictionary')
    calculateError(non_melanomas, D_magnitude, 'magnitude')
    return

    print('Color')
    print('melanomas with melanoma dictionary')
    calculateError(melanomas, D_m_color, 'color')
    print('non melanomas with melanoma dictionary')
    calculateError(non_melanomas, D_m_color, 'color')
    print('melanomas with non melanoma dictionary')
    calculateError(melanomas, D_n_color, 'color')
    print('non melanomas with non melanoma dictionary')
    calculateError(non_melanomas, D_n_color, 'color')
    print('melanomas with full dictionary')
    calculateError(melanomas, D_color, 'color')
    print('non melanomas with full dictionary')
    calculateError(non_melanomas, D_color, 'color')

    print('Magnitude')
    print('melanomas with melanoma dictionary')
    calculateError(melanomas, D_m_magnitude, 'magnitude')
    print('non melanomas with melanoma dictionary')
    calculateError(non_melanomas, D_m_magnitude, 'magnitude')
    print('melanomas with non melanoma dictionary')
    calculateError(melanomas, D_n_magnitude, 'magnitude')
    print('non melanomas with non melanoma dictionary')
    calculateError(non_melanomas, D_n_magnitude, 'magnitude')
    print('melanomas with full dictionary')
    calculateError(melanomas, D_magnitude, 'magnitude')
    print('non melanomas with full dictionary')
    calculateError(non_melanomas, D_magnitude, 'magnitude')

    return
    train_color_histograms = getHistograms(train_dataset, atoms_c, D_color, 'color')
    train_magnitude_histograms = getHistograms(train_dataset, atoms_m, D_magnitude, 'magnitude')

    ####################### GET TEST DATA #################################

    #test_labels = getLabels('Data/ISIC-2017_Test_v2_Part3_GroundTruth.csv')
    #test_dataset = getData('testData', test_labels, False)

    #filename = 'saves/test_features_overlapp.sav'
    #joblib.dump(test_dataset, filename)

    test_dataset = joblib.load('saves/datasets/test_dataset.sav')
    test_labels = labelsToArray(test_dataset)

    ####################### GET TEST HISTOGRAMS ############################

    test_color_histograms = getHistograms(test_dataset, atoms_c, D_color, 'color')
    test_magnitude_histograms = getHistograms(test_dataset, atoms_m, D_magnitude, 'magnitude')

    #for c_c in C:
    #    for c_g in gamma:
    #        for m_c in C:
    #            for m_g in gamma:
    ####################### TRAIN CLASSIFIERS ############################

    color_classifier = getClassifier(train_color_histograms, np.ravel(train_labels), 'SVM', c_c, c_g, c_w)
    magnitude_classifier = getClassifier(train_magnitude_histograms, np.ravel(train_labels), 'SVM', m_c, m_g, m_w)

    ####################### GET RESULTS ###################################
    #print('For color C=',c_c,'; color gamma=',c_g,'mag C=',m_c,';mag gamma=',m_g)
    result = predict2Classifiers(color_classifier,
                        magnitude_classifier,
                        test_color_histograms, test_magnitude_histograms, test_labels)


main()
