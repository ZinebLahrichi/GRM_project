from tools import *

from utils import create_scribble, compute_distance, compute_segmentation
from utils import open_img, label2rgb, read_ground_truth
from default_parameters import *

script_dir = os.path.dirname(__file__)
input_dir = os.path.join(script_dir, 'input/')

alphas = [0.0001, 0.1, 10, 1000]

if __name__ == '__main__':

    img_name = 'cow'

    img = open_img(input_dir + img_name + '.jpg', default_shape)
    img = img / np.max(img) * 255.
    NB_CLASSES = 6

    scribbles, X, Y, I = create_scribble(img, img_name, input_dir, showScribble, NB_CLASSES)

    print('>> Computing shortest path to scribble...')
    dists = compute_distance(img, X, Y, NB_CLASSES, showDistanceMap)

    bests = []
    for alpha in alphas:
        print('>> Computing segmentation...')
        best = compute_segmentation(img, scribbles, X, Y, I, dists, alpha, eps, sigma, USE_DIST, USE_COLOR, MARGINAL,
                                    NB_CLASSES, NB_ITERS, LMBD)
        bests.append(best)

    ground_truth_filepath = input_dir + img_name + '_layers.txt'
    gt = read_ground_truth(ground_truth_filepath, default_shape)

    plt.figure(figsize=(20, 15))
    fig, axes = plt.subplots(1, len(alphas) + 1)
    for ax, best, alpha in zip(axes, bests, alphas):
        ax.imshow(label2rgb(best, NB_CLASSES))
        ax.title.set_text(f'alpha = {alpha}')
    axes[-1].imshow(label2rgb(gt, NB_CLASSES))
    plt.show()



