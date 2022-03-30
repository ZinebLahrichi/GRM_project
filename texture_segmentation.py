from tqdm import tqdm

from tools import *

from utils_texture import compute_texture_info, compute_segmentation, create_scribble, compute_distance, plot_segmentation_and_image



script_dir = os.path.dirname(__file__)
input_dir = os.path.join(script_dir, 'input/')

###########################################################################################
# Parameters of the experiment
# Baseline: alpha=30, sigma=30, LMBD=100, NB_ITERS = 100

showScribble =  True
showDistanceMap = True
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

    img_name = 'car'

    original_img = plt.imread(input_dir + img_name + '.jpg')

    img = compute_texture_info(original_img)
    print(f"image with texture shape: {img.shape}")


    img = img / np.max(img) * 255.
    NB_CLASSES = 2

    scribbles, X,Y,I = create_scribble(img, img_name, input_dir, showScribble, NB_CLASSES,color_palette)


    # Compute shortest path to scribble (rho)
    print('>> Computing shortest path to scribble...')
    n, m = img.shape[:2]
    dists  = compute_distance(img, X, Y, NB_CLASSES, showDistanceMap)

    print('>> Computing segmentation...')
    best = compute_segmentation(img, scribbles, X, Y, I, dists, alpha, eps, sigma, USE_DIST, USE_COLOR, MARGINAL, NB_CLASSES, NB_ITERS, LMBD)
    print('... Done')


    # Display results

    if showSegmentation:
        plt.figure(figsize=(14, 12))
        plot_segmentation_and_image(best, original_img, 2)
