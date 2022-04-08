from tools import *

from utils import create_scribble, compute_distance, compute_segmentation
from utils import open_img, label2rgb, read_ground_truth
from default_parameters import *

script_dir = os.path.dirname(__file__)
input_dir = os.path.join(script_dir, 'input/')

sigmas = [1, 30,  100, 500]

if __name__ == '__main__':

    img_name = 'cow'

    img = open_img(input_dir + img_name + '.jpg', down_shape)
    img = img / np.max(img) * 255.
    NB_CLASSES = 6

    scribbles, X, Y, I = create_scribble(img, img_name, input_dir, showScribble, NB_CLASSES)

    print('>> Computing shortest path to scribble...')
    dists = compute_distance(img, X, Y, NB_CLASSES, showDistanceMap)

    bests = []
    for sigma in sigmas:
        print('>> Computing segmentation...')
        best = compute_segmentation(img, scribbles, X, Y, I, dists, alpha, eps, sigma, USE_DIST, USE_COLOR, MARGINAL,
                                    NB_CLASSES, NB_ITERS, LMBD)
        bests.append(best)

    ground_truth_filepath = input_dir + img_name + '_layers.txt'
    gt = read_ground_truth(ground_truth_filepath, default_shape)

    plt.figure(figsize=(20, 15))
    fig, axes = plt.subplots(1, len(sigmas) + 1)
    for ax, best, sigma in zip(axes, bests, sigmas):
        ax.imshow(label2rgb(best, NB_CLASSES))
        ax.title.set_text(f'sigma = {sigma}')

    axes[-1].imshow(label2rgb(gt, NB_CLASSES))
    plt.show()



