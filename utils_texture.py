import cv2
import numpy as np
from tools import *
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from default_parameters import color_palette


def build_gabor_kernels(number_of_orientations = 10):
    # Building some gabor kernels to filter image

    filters = []
    ksize = 40
    if number_of_orientations!=0:
        orientations = [i*int(360/number_of_orientations) for i in range(number_of_orientations)]
        for rotation in orientations:
            kernel = cv2.getGaborKernel((ksize, ksize), 4.25, rotation,3, 0.5, 0, ktype=cv2.CV_32F)
            filters.append(kernel)
        return filters
    else:
        return([])


def compute_texture_info(image, number_of_texture_filters_to_use = 10):
    """
    return an image with the first 3 channels corresponding to RGB channels
    and extra 9 channels for texture information
    """
    rows, cols, channels = image.shape
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gaborKernels = build_gabor_kernels(number_of_texture_filters_to_use)
    texture = np.zeros((rows, cols, len(gaborKernels)+3))

    for (i, kernel) in enumerate(gaborKernels):
        filteredImage = cv2.filter2D(gray, cv2.CV_8UC1, kernel)    
        texture[:,:,3+i] = filteredImage

    # add rgb channels
    texture[:,:,:3] = image
    return(texture)




def create_scribble(img, img_name, input_dir, showScribble, NB_CLASSES,color_palette):
    scribbles = np.zeros(img.shape[:2])
    for i in range(1, NB_CLASSES + 1):
        img_scribble = plt.imread(input_dir + img_name + '_' + str(i) + '.png')
        img_scribble = np.mean(img_scribble, axis=2)
        scribbles[img_scribble < 0.50] = i

    X = []  # coordinates scribble points for each label (x axis)
    Y = []  # coordinates scribble points for each label (y axis)
    I = []  # intensity (array [R, G, B, texture channels]) --> 13 channels in total

    for i in range(1, NB_CLASSES + 1):
        Xi, Yi = np.where(scribbles == i)
        Ii = np.array([img[Xi[k], Yi[k]] for k in range(Xi.shape[0])])
        X.append(Xi)
        Y.append(Yi)
        I.append(Ii)

    # Show 3 classes-scribbles
    if showScribble:
        plt.figure()
        img2 = np.zeros((img.shape[0], img.shape[1], 3))
        for i in range(1, NB_CLASSES + 1):
            img2[scribbles == i, :] = color_palette[i]
        plt.imshow(img2)

        plt.title("classes-scribbles")
        plt.show()

    return scribbles, X,Y,I


def compute_distance(img, X, Y, NB_CLASSES, showDistanceMap):
    n, m = img.shape[:2]
    dists = np.zeros((NB_CLASSES, n, m))

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(NB_CLASSES):
                dists[k, i, j] = np.min(np.power((X[k] - i) / n, 2) + np.power((Y[k] - j) / m, 2))
    dists = np.sqrt(dists)

    # Display distance map
    if showDistanceMap:
        plt.figure()
        for k in range(NB_CLASSES):
            plt.subplot(1, NB_CLASSES, k + 1)
            plt.imshow(dists[k, :, :])
        plt.show()
    return dists



def compute_probability(img, scribbles, X, Y, I, dists, alpha, eps, sigma, USE_DIST, USE_COLOR, MARGINAL, NB_CLASSES):
    seg_class = np.zeros(img.shape[:2])
    P = np.zeros((NB_CLASSES, img.shape[0], img.shape[1]))

    # Compute the joint probability distribution for all classes
    # Loop over the patches of the image
    for i in tqdm(range(0, img.shape[0])):
        for j in range(0, img.shape[1]):
            # If (i, j) already has a label
            if np.sum(scribbles[i, j]) != 0:
                pass
            for k in range(NB_CLASSES):

                rho = alpha * dists[k, i, j]
                if USE_DIST and rho != 0:
                    p_dist = (1 / np.sqrt(2 * np.pi * rho ** 2)) * np.exp(-(np.power((X[k] - i) / img.shape[0], 2)
                                                                            + np.power((Y[k] - j) / img.shape[1],
                                                                                       2)) / (2 * rho ** 2))
                else:
                    p_dist = np.ones(X[k].shape)

                if USE_COLOR:
                    p_col = (1 / np.power(2 * np.pi * sigma ** 2, 1.5)) * np.exp(
                        -np.sum(np.power(I[k] - img[i, j], 2), axis=1) / (2 * sigma ** 2))
                else:
                    p_col = np.ones(X[k].shape)

                m = X[k].shape[0]
                p = np.dot(p_col, p_dist) / m

                P[k, i, j] = p

    # Compute the marginal distribution by normalizing
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


    return seg_class, P, f

def compute_segmentation(img, scribbles, X, Y, I, dists, alpha, eps, sigma, USE_DIST, USE_COLOR, MARGINAL, NB_CLASSES, NB_ITERS, LMBD):

    seg_class, P, f = compute_probability(img, scribbles, X, Y, I, dists, alpha, eps, sigma, USE_DIST, USE_COLOR, MARGINAL, NB_CLASSES)

    # Run the optimization
    g = compute_g(img)
    smooth_seg_class = np.around(gaussian_filter(seg_class, 1))
    theta_ini = compute_theta(smooth_seg_class)
    primal, dual, hist_theta = optimize_primal_dual(f, g, NB_ITERS, theta_ini, lmbd=LMBD)

    return  np.argmax(hist_theta[-1], axis=0)


def label2rgb(labels, n_class):
    img2 = np.zeros((labels.shape[0], labels.shape[1], 3))
    for i in range(0, n_class):
        img2[labels == i, :] = color_palette[i]
    return img2


def plot_segmentation(labels, n_class):
    img = label2rgb(labels, n_class)
    plt.imshow(img)
    plt.show()

    
def plot_segmentation_and_image(labels, original_image, n_class):
    img = label2rgb(labels, n_class)
    plt.imshow(original_image)
    plt.imshow(img, alpha = 0.5)
    plt.show()