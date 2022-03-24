from tools import *

from utils import create_scribble, compute_distance, compute_segmentation
from utils import open_img, read_ground_truth, label2rgb
from default_parameters import *
from skimage.restoration import denoise_bilateral, denoise_wavelet


script_dir = os.path.dirname(__file__)
input_dir = os.path.join(script_dir, 'input/')

if __name__ == '__main__':

    img_name = 'cow'

    img = open_img(input_dir + img_name + '.jpg', down_shape)
    denoised_wavelet = denoise_wavelet(img, channel_axis=-1, sigma=0.2)
    denoised_bilateral = denoise_bilateral(img, channel_axis=-1, sigma_color=5, sigma_spatial=2)
    smoothed_images = [denoised_wavelet, denoised_bilateral]
    denoising_methods = ['wavelet', 'bilateral']
    NB_CLASSES = 6

    bests = []
    for img in smoothed_images:
        img = img / np.max(img) * 255.

        scribbles, X, Y, I = create_scribble(img, img_name, input_dir, showScribble, NB_CLASSES)

        print('>> Computing shortest path to scribble...')
        n, m = img.shape[:2]
        dists = compute_distance(img, X, Y, NB_CLASSES, showDistanceMap)

        print('>> Computing segmentation...')
        best = compute_segmentation(img, scribbles, X, Y, I, dists, alpha, eps, sigma, USE_DIST, USE_COLOR, MARGINAL,
                                    NB_CLASSES, NB_ITERS, LMBD)
        bests.append(best)

    ground_truth_filepath = input_dir + img_name + '_layers.txt'
    gt = read_ground_truth(ground_truth_filepath, default_shape)

    plt.figure(figsize=(20, 15))
    fig, axes = plt.subplots(1, len(smoothed_images))
    for ax, best, method in zip(axes, bests, denoising_methods):
        ax.imshow(label2rgb(best, NB_CLASSES))
        ax.title.set_text(method)
    axes[-1].imshow(label2rgb(gt, NB_CLASSES))
    plt.show()




