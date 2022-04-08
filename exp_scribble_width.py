from tools import *

from utils import create_scribble, compute_distance, compute_segmentation
from utils import open_img, read_ground_truth, label2rgb
from default_parameters import *

script_dir = os.path.dirname(__file__)
input_dir = os.path.join(script_dir, 'input/')

widths = [0.2, 0.5, 0.9]

if __name__ == '__main__':

    img_name = 'cow'
    showScribble = False
    default_shape = open_img(input_dir + img_name + '.jpg', None).shape[:2]
    img = open_img(input_dir + img_name + '.jpg', default_shape)
    img = img / np.max(img) * 255.
    NB_CLASSES = 6

    bests = []
    for width in widths:
        scribbles, X, Y, I = create_scribble(img, img_name, input_dir, showScribble, NB_CLASSES, width=width)

        print('>> Computing shortest path to scribble...')
        dists = compute_distance(img, X, Y, NB_CLASSES, showDistanceMap)

        print('>> Computing segmentation...')
        best = compute_segmentation(img, scribbles, X, Y, I, dists, alpha, eps, sigma, USE_DIST, USE_COLOR, MARGINAL,
                                    NB_CLASSES, NB_ITERS, LMBD)
        bests.append(best)

    ground_truth_filepath = input_dir + img_name + '_layers.txt'
    gt = read_ground_truth(ground_truth_filepath, default_shape)

    plt.figure(figsize=(20, 15))
    fig, axes = plt.subplots(1, len(widths) + 1)
    for ax, best, width in zip(axes, bests, widths):

        segmented = label2rgb(best, NB_CLASSES)
        ax.imshow(segmented)
        ax.title.set_text(f'width: {width}')

    axes[-1].imshow(label2rgb(gt, NB_CLASSES))
    plt.show()






