from tools import *

from utils import create_scribble, compute_distance
from utils import open_img, plot_segmentation, plot_segmentation_and_gt, read_ground_truth, compute_segmentation
from default_parameters import *

script_dir = os.path.dirname(__file__)
input_dir = os.path.join(script_dir, 'input/')

if __name__ == '__main__':

    img_name = 'cow'

    img = open_img(input_dir + img_name + '.jpg', down_shape)
    img = img / np.max(img) * 255.
    NB_CLASSES = 6

    scribbles, X, Y, I = create_scribble(img, img_name, input_dir, showScribble, NB_CLASSES)
    # Compute shortest path to scribble (rho)
    print('>> Computing shortest path to scribble...')
    dists = compute_distance(img, X, Y, NB_CLASSES, showDistanceMap)
    print('>> Computing segmentation...')
    best = compute_segmentation(img, scribbles, X, Y, I, dists, alpha, eps, sigma, USE_DIST, USE_COLOR, MARGINAL, NB_CLASSES, NB_ITERS, LMBD)
    print('... Done')

    if showSegmentation:
        # Final segmentation

        if showGroundTruth:
            ground_truth_filepath = input_dir + img_name + '_layers.txt'
            gt = read_ground_truth(ground_truth_filepath, default_shape)
            plot_segmentation_and_gt(best, gt, NB_CLASSES)

        else:
            plt.figure(figsize=(14, 12))
            plot_segmentation(best, NB_CLASSES)


