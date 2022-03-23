from tqdm import tqdm

from tools import *
from scipy.ndimage import gaussian_filter
from utils_video import compute_probability, create_scribble, compute_distance, get_XYIT
import argparse
from os import listdir
from os.path import isfile, join

script_dir = os.path.dirname(__file__)
input_dir = os.path.join(script_dir, 'DAVIS\\JPEGImages\\480p\\')

############################################################################################
# Add Parser

parser = argparse.ArgumentParser(description='Arguments for the video tracking')
parser.add_argument('img_name', type=str,
                    help='Name of the video/image to process. If video, points toward the directory where the images are')
parser.add_argument('--NB_CLASSES', type=int, default= 2,
                    help='Number of classes to detect')


###########################################################################################
# Parameters of the experiment
# Baseline: alpha=30, sigma=30, LMBD=100, NB_ITERS = 100

showScribble = False
showDistanceMap = True
showSegmentation = True
NB_ITERS = 50
alpha = 30  # Distance parameter: indicates how segmentation is locally sensible to the input scribble
sigma = 30  # Smoothing parameter: indicates how segmentation depends on color
tau = 5  # Memory parameter : indicates how segmentation depends on previous frames
LMBD = 30  # Regularization parameter:
eps = 10e-7
USE_DIST = True
USE_COLOR = True
MARGINAL = False
color_palette = [[0, 1, 1], [1, 0, 0], [0, 0, 1], [0, 1, 0], [1, 1, 1], [0.5, 0.5, 0]]
###########################################################################################

def get_video_info(onlyfiles):
    ind_scribbles = []
    frames = []
    for f in onlyfiles:
        if '_' in f:
            ind_scribbles.append(f[:5])
        else:
            frames.append(f[:5])
    return ind_scribbles, frames


if __name__ == '__main__':
    args = parser.parse_args()


    img_name = args.img_name
    NB_CLASSES = args.NB_CLASSES

    onlyfiles = [f for f in listdir(input_dir + img_name) if isfile(join(input_dir + img_name, f))]
    ind_scribbles, frames =get_video_info(onlyfiles)

    X,Y,I,T = get_XYIT(ind_scribbles, input_dir, img_name, NB_CLASSES)

    Final_images = []

    for t in frames:
        img = plt.imread(input_dir + img_name + t + '.jpg')
        img = img / np.max(img) * 255.
        img_t_name = input_dir + img_name + t
        scribbles = create_scribble(img, img_name, input_dir, showScribble, NB_CLASSES,color_palette)

        # Compute shortest path to scribble (rho)
        print('>> Computing shortest path to scribble...')
        n, m = img.shape[:2]
        dists  = compute_distance(img, X, Y, NB_CLASSES, showDistanceMap)

        print('>> Computing segmentation...')

        seg_class,P,f = compute_probability(img, scribbles, t, X, Y, I, T, dists, alpha, eps, sigma, tau, USE_DIST, USE_COLOR, MARGINAL, NB_CLASSES)

        # Run the optimization
        g = compute_g(img)
        smooth_seg_class = np.around(gaussian_filter(seg_class, 1))
        theta_ini = compute_theta(smooth_seg_class)
        primal, dual, hist_theta = optimize_primal_dual(f, g, NB_ITERS, theta_ini, lmbd=LMBD)

        print('... Done')

        # Display results
        best = np.argmax(hist_theta[-1], axis=0)
        print(best)
        plt.figure(figsize=(14, 12))

        # plt.subplot(1, NB_CLASSES, 1)
        img2 = np.zeros(img.shape)
        for i in range(0, NB_CLASSES):
            img2[best == i, :] = color_palette[i]
        Final_images.append(img2)

        if showSegmentation:

            # Final segmentation
            plt.imshow(img2)
            plt.show()