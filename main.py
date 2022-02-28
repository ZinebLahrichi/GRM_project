# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from tools import *
from scipy.ndimage.filters import gaussian_filter
from array2gif import write_gif

# Paths
script_dir = os.path.dirname(__file__)
input_dir = os.path.join(script_dir, 'input/')

# Parameters of the experiment
showPlot = False
NB_ITERS = 100
alpha = 30
step = 1
sigma = 30
eps = 10e-7
USE_DIST = True
USE_COLOR = True
MARGINAL = False


if __name__ == '__main__':
            
    img_name = 'car'

    img = plt.imread(input_dir + img_name + '.jpg')
    img = img/np.max(img)*255.

    NB_CLASSES = 2

    ##

    scribbles = np.zeros(img.shape[:2])

    FROM_FILE = True
    MARGINAL = False

    if not FROM_FILE:
        # for quick experiments
        scribbles[100, 100] = 1
        scribbles[100, 200] = 1
        scribbles[200, 200] = 2
    else:
        for i in range(1, NB_CLASSES + 1):        
            img_scrb = plt.imread(input_dir + img_name + '_' + str(i) + '.png')[:, :, 0]
            print(img_scrb)
            scribbles[img_scrb == 0] = i
            print("oui")
            print(scribbles)
    X = []
    Y = []
    I = []

    for i in range(1, NB_CLASSES + 1):
        Xi, Yi = np.where(scribbles == i)
        Ii = np.array([img[Xi[k], Yi[k]] for k in range(Xi.shape[0])])
        X.append(Xi)
        Y.append(Yi)
        I.append(Ii)

    ## show 3 classes-scribbles
    if showPlot:
        plt.figure()
        img2 = np.copy(img)
        img2[scribbles == 1, :] = [1, 0, 0]
        img2[scribbles == 2, :] = [0, 1, 0]
        img2[scribbles == 3, :] = [0, 0, 1]

        plt.imshow(img2)
        plt.show()


    ## compute shortest path to scribble ~ 10s

    print('>> Computing shortest path to scribble...')
    n, m = img.shape[:2]
    dists = np.zeros((NB_CLASSES, n, m))

    nb_scribbles = [np.count_nonzero(scribbles==k+1) for k in range(NB_CLASSES)]

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(NB_CLASSES):
                dists[k, i, j] = np.min(np.power((X[k] - i)/n, 2) + np.power((Y[k] - j)/m, 2))
    dists = np.sqrt(dists)

    ## pour le style, display distance map

    if showPlot:
        plt.figure()
        for k in range(NB_CLASSES):
            plt.subplot(1, NB_CLASSES, k+1)
            plt.imshow(dists[k, :, :])
        plt.show()


    ## segmentation ~ long et pas optimal...

    print('>> Computing segmentation...')

    seg_class = np.zeros(img.shape[:2])
    P = np.zeros((NB_CLASSES, img.shape[0], img.shape[1]))

    ### Compute the joint probabiliy distribution for all classes
    # Loop over the patches of the image
    for i in tqdm(range(0, img.shape[0])):
        for j in range(0, img.shape[1]):
            
            if np.sum(scribbles[i, j]) != 0: # Si on est sur un scribble
                pass
        
            for k in range(NB_CLASSES):

                rho = alpha * dists[k, i, j]
                if USE_DIST and rho!= 0:
                    p_dist = (1 / np.sqrt(2 * np.pi * rho**2)) * np.exp(-(np.power((X[k] - i)/ img.shape[0], 2)
                                                                          +  np.power((Y[k] - j)/ img.shape[1], 2)) / (2 * rho**2))
                else:
                    p_dist = np.ones(X[k].shape)
        
                if USE_COLOR:
                    p_col = (1 / np.power(2 * np.pi * sigma**2, 1.5)) * np.exp(-np.sum(np.power(I[k] - img[i,j], 2), axis = 1) / (2 * sigma**2))
                else:
                    p_col = np.ones(X[k].shape)

                m = X[k].shape[0]
                p = np.dot(p_col, p_dist) / m

                P[k, i, j] = p

    ### Compute the marginal distribution by normalizing
    if MARGINAL:
        for k in range(NB_CLASSES):
            P /= np.mean(P, axis=0)

    f = np.zeros(P.shape)
    for k in range(NB_CLASSES):
        min_P = np.min(P[k])
        max_P = np.max(P[k])
        f[k] = - np.log((P[k] - min_P) / (max_P - min_P) * (1 - eps) + eps)

    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            class_pixel = int(scribbles[i, j] - 1)
            if class_pixel >= 0:
                f[:, i, j] = 10e7
                f[class_pixel, i, j] = 0

    # For each patch, choose the class that has the highest probability
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            seg_class[i, j] = np.argmax(P[:, i, j])


    ### Run the optimization
    g = compute_g(img)
    smooth_seg_class = np.around(gaussian_filter(seg_class, 1))
    theta_ini = compute_theta(smooth_seg_class)
    primal, dual, hist_theta = optimize_primal_dual(f, g, 100, theta_ini)

    print('... Done')

    ## Display results
    if showPlot:
        # Evolution of the energys and gap
        plot_results(primal, dual)

        # Final segmentation
        best = np.argmax(hist_theta[-1], axis=0)

        plt.figure(figsize=(14,12))

        plt.subplot(1, NB_CLASSES, 1)
        img2 = np.copy(img)
        img2[best == 0, :] = [0.8, 0, 0.2]
        plt.imshow(img2)

        plt.subplot(1, NB_CLASSES, 2)
        img2 = np.copy(img)
        img2[best == 1, :] = [0, 0.8, 0.2]
        plt.imshow(img2)
        
        plt.show()