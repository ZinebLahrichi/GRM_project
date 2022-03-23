from cv2 import imread
from tools import *
from tqdm import tqdm
from os.path import isfile
from skimage.transform import resize

shape = (240,480)

def open_img(path, shape = shape):
    img = imread(path)
    if shape == None:
        return img
    return resize(img, shape)


def get_XYIT(ind_scribbles, input_dir, img_name, NB_CLASSES):
    X = []  # coordinates scribble points for each label (x axis)
    Y = []  # coordinates scribble points for each label (y axis)
    I = []  # intensity (array [R, G, B])
    T = []  # frame
    for _ in range(NB_CLASSES):
        X.append(np.array([]))

        Y.append(np.array([]))
        I.append(np.array([]))
        T.append(np.array([]))
    for f in ind_scribbles:
        img = open_img(input_dir + img_name + f + '.jpg')
        scribbles = np.zeros(img.shape[:2])
        for i in range(1, NB_CLASSES + 1):
            img_scribble = open_img(input_dir + img_name + f + '_' + str(i) + '.png')
            img_scribble = np.mean(img_scribble, axis=2)
            scribbles[img_scribble < 0.50] = i


        for i in range(NB_CLASSES):
            Xi, Yi = np.where(scribbles == i+1)
            Ii = np.array([img[Xi[k], Yi[k]] for k in range(Xi.shape[0])])
            X[i] = np.concatenate([X[i].reshape(-1) , Xi])
            Y[i]= np.concatenate([Y[i].reshape(-1) , Yi])
            I[i]= np.concatenate([I[i].reshape(-1,3) , Ii])
            T[i]= np.concatenate([T[i].reshape(-1) , int(f) * np.ones(Xi.shape[0])])

    return X,Y,I,T



def create_scribble(img, img_name, input_dir, showScribble, NB_CLASSES,color_palette):
    scribbles = np.zeros(img.shape[:2])
    if isfile(input_dir + img_name + '_' + str(1) + '.png'):
        for i in range(1, NB_CLASSES + 1):
            img_scribble = open_img(input_dir + img_name + '_' + str(i) + '.png')
            img_scribble = np.mean(img_scribble, axis=2)
            scribbles[img_scribble < 0.50] = i

    # Show 3 classes-scribbles
    if showScribble:
        plt.figure()
        img2 = np.zeros(img.shape)
        for i in range(1, NB_CLASSES + 1):
            img2[scribbles == i, :] = color_palette[i]
        plt.imshow(img2)

        plt.title("classes-scribbles")
        plt.show()

    return scribbles


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



def compute_probability(img, scribbles, t, X, Y, I, T, dists, alpha, eps, sigma, tau, USE_DIST, USE_COLOR, MARGINAL, NB_CLASSES):
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
                    p_dist = (1 / np.sqrt(2 * np.pi * rho ** 2))
                    p_dist *= np.exp(-(np.power((X[k] - i) / img.shape[0], 2)
                                + np.power((Y[k] - j) / img.shape[1],
                                2)) / (2 * rho ** 2)
                            )
                else:
                    p_dist = np.ones(X[k].shape)

                if USE_COLOR:
                    p_col = (1 / np.power(2 * np.pi * sigma ** 2, 1.5)) * np.exp(
                        -np.sum(np.power(I[k] - img[i, j], 2), axis=1) / (2 * sigma ** 2))
                else:
                    p_col = np.ones(X[k].shape)
                
                p_time = np.diag(
                    (1 / np.power(2 * np.pi * tau ** 2, 1.5)) * np.exp(
                        -np.power(T[k] - int(t), 2) / (2 * tau ** 2))
                )


                m = X[k].shape[0]
                p = np.dot(p_col, p_time @ p_dist) / m

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