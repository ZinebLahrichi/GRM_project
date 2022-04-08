from tools import *

from utils import create_scribble, compute_distance, compute_segmentation
from utils import open_img, read_ground_truth, label2rgb
from default_parameters import *
from skimage.restoration import denoise_bilateral


script_dir = os.path.dirname(__file__)
input_dir = os.path.join(script_dir, 'input/')

if __name__ == '__main__':

    img_name = 'boat'
    NB_CLASSES = 9

    NB_ITERS = 10
    LMBD = 0.1
    alpha = 100
    sigma = 40

    default_shape = open_img(input_dir + img_name + '.jpg', None).shape[:2]
    img = open_img(input_dir + img_name + '.jpg', down_shape)
    denoised_bilateral = denoise_bilateral(img, channel_axis=-1, sigma_color=1, sigma_spatial=1, mode='edge')
    denoising_methods = ['original', 'pre-denoising']
    smoothed_images = [img, denoised_bilateral]

    bests = []
    for img2 in smoothed_images:
        img2 = img2 / np.max(img2) * 255.

        scribbles, X, Y, I = create_scribble(img2, img_name, input_dir, showScribble, NB_CLASSES)

        print('>> Computing shortest path to scribble...')
        dists = compute_distance(img2, X, Y, NB_CLASSES, showDistanceMap)

        print('>> Computing segmentation...')
        best = compute_segmentation(img2, scribbles, X, Y, I, dists, alpha, eps, sigma, USE_DIST, USE_COLOR, MARGINAL,
                                    NB_CLASSES, NB_ITERS, LMBD)
        bests.append(best)

    ground_truth_filepath = input_dir + img_name + '_layers.txt'
    gt = read_ground_truth(ground_truth_filepath, default_shape)

    plt.figure(figsize=(20, 15))
    fig, axes = plt.subplots(len(smoothed_images) + 1, 3)

    for ax, best, method, original_img in zip(axes, bests, denoising_methods, smoothed_images):

        ax[0].imshow(original_img)
        ax[0].title.set_text(method)

        segmented = label2rgb(best, NB_CLASSES)
        ax[1].imshow(segmented)
        ax[1].title.set_text(method + ' + segmentation')

    axes[-1][0].imshow(label2rgb(gt, NB_CLASSES))
    axes[-1][1].imshow(label2rgb(gt, NB_CLASSES))
    plt.show()






