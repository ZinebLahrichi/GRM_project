from tqdm import tqdm

from tools import *
from scipy.ndimage import gaussian_filter

from utils import compute_probability, create_scribble, compute_distance



script_dir = os.path.dirname(__file__)
input_dir = os.path.join(script_dir, 'input/')

###########################################################################################
# Parameters of the experiment
# Baseline: alpha=30, sigma=30, LMBD=100, NB_ITERS = 100

showScribble = False
showDistanceMap = False
showSegmentation = True
NB_ITERS = 100
alpha = 30  # Distance parameter: indicates how segmentation is locally sensible to the input scribble
sigma = 30  # Smoothing parameter: indicates how segmentation depends on color
LMBD = 30  # Regularization parameter:
eps = 10e-7
USE_DIST = True
USE_COLOR = True
MARGINAL = False
color_palette = [[0, 1, 1], [1, 0, 0], [0, 0, 1], [0, 1, 0], [1, 1, 1], [0.5, 0.5, 0]]
###########################################################################################

if __name__ == '__main__':

    img_name = 'chat'

    img = plt.imread(input_dir + img_name + '.jpg')
    img = img / np.max(img) * 255.
    NB_CLASSES = 2

    scribbles, X,Y,I = create_scribble(img, img_name, input_dir, showScribble, NB_CLASSES,color_palette)

    # Compute shortest path to scribble (rho)
    print('>> Computing shortest path to scribble...')
    n, m = img.shape[:2]
    dists  = compute_distance(img, X, Y, NB_CLASSES, showDistanceMap)

    print('>> Computing segmentation...')

    seg_class,P,f = compute_probability(img, scribbles, X, Y, I, dists, alpha, eps, sigma, USE_DIST, USE_COLOR, MARGINAL, NB_CLASSES)


    # Run the optimization
    g = compute_g(img)
    smooth_seg_class = np.around(gaussian_filter(seg_class, 1))
    theta_ini = compute_theta(smooth_seg_class)
    primal, dual, hist_theta = optimize_primal_dual(f, g, NB_ITERS, theta_ini, lmbd=LMBD)

    print('... Done')

    # Display results

    if showSegmentation:

        # Final segmentation
        best = np.argmax(hist_theta[-1], axis=0)
        print(best)
        plt.figure(figsize=(14, 12))

        # plt.subplot(1, NB_CLASSES, 1)
        img2 = np.zeros(img.shape)
        for i in range(0, NB_CLASSES):
            img2[best == i, :] = color_palette[i]

        plt.imshow(img2)
        plt.show()
