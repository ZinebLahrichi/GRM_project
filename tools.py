import numpy as np
import matplotlib.pyplot as plt
import os
from array2gif import write_gif


def double(x):
    """
        Trick to duplicate the array x
        Input shape is x.shape
        Output shape is (*x.shape, 2)
    """
    return x.dot(np.ones((x.shape[1], x.shape[1], 2)))


def compute_g(img, gamma=5):
    """
        Computes the function g(x) = exp(-gamma * abs(grad(I)))
    """

    h, w, _ = img.shape
    edge = np.zeros((h, w))

    for color in range(3):
        grad = forward_gradient(img[:, :, color])
        edge += np.linalg.norm(grad, ord=2, axis=2) ** 2

    g = np.exp(-gamma * np.sqrt(edge))

    return g


def forward_gradient(im):
    """
    Function to compute the forward gradient of the image I.
    Definition from: http://www.ipol.im/pub/art/2014/103/, p208
    :param im: numpy array [MxN], input image
    :return: numpy array [MxNx2], gradient of the input image, the first channel is the horizontal gradient, the second
    is the vertical gradient.
    """

    h, w = im.shape
    gradient = np.zeros((h, w, 2), im.dtype)  # Allocate gradient array
    # Horizontal direction
    gradient[:, :-1, 0] = im[:, 1:] - im[:, :-1]
    # Vertical direction
    gradient[:-1, :, 1] = im[1:, :] - im[:-1, :]

    return gradient


def backward_divergence(grad):
    """
    Function to compute the backward divergence.
    Definition in : http://www.ipol.im/pub/art/2014/103/, p208
    ## :param grad: numpy array [NxMx2], array with the same dimensions as the gradient of the image to denoise.
    :return: numpy array [NxM], backward divergence
    """

    h, w = grad.shape[:2]
    div = np.zeros((h, w), grad.dtype)  # Allocate divergence array
    # Horizontal direction
    d_h = np.zeros((h, w), grad.dtype)
    d_h[:, 0] = grad[:, 0, 0]
    d_h[:, 1:-1] = grad[:, 1:-1, 0] - grad[:, :-2, 0]
    d_h[:, -1] = -grad[:, -2:-1, 0].flatten()

    # Vertical direction
    d_v = np.zeros((h, w), grad.dtype)
    d_v[0, :] = grad[0, :, 1]
    d_v[1:-1, :] = grad[1:-1, :, 1] - grad[:-2, :, 1]
    d_v[-1, :] = -grad[-2:-1, :, 1].flatten()

    # Divergence
    div = d_h + d_v
    return div


def compute_dual_energy(f, div, lmbd):
    nb_classes = f.shape[0]
    mini = np.zeros(f.shape)

    for k in range(nb_classes):
        mini[k] = (1 / lmbd) * f[k] - div[k]

    return np.sum(np.min(mini, axis=0))


def compute_primal_energy(theta, f, g, lmbd):
    nb_classes = f.shape[0]

    grad = [forward_gradient(theta[k]) for k in range(nb_classes)]
    norm_grad = np.linalg.norm(grad, axis=3)
    energy = np.sum([(1 / lmbd) * theta[k] * f[k] - norm_grad[k] * g
                     for k in range(nb_classes)])

    return energy


def compute_theta(seg_class):
    nb_classes = len(np.unique(seg_class))
    theta = np.zeros((nb_classes, *seg_class.shape))

    for k in range(nb_classes):
        theta[k] = (seg_class == k) * 1

    return theta


def save_gif(hist_theta, fps=10):
    images = [np.argmax(t, axis=0) * 255 for t in hist_theta]
    dataset = [np.array([t, t, t]) for t in images]

    script_dir = os.path.dirname(__file__)
    results_dir = os.path.join(script_dir, 'output/')
    write_gif(np.clip(dataset, 0, 255), results_dir + 'result.gif', fps=fps)


def euclidean_proj_simplex(v, s=1):
    """ Compute the Euclidean projection on a positive simplex
        Solves the optimisation problem (using the algorithm from [1]):
        min_w 0.5 * || w - v ||_2^2 , s.t. \sum_i w_i = s, w_i >= 0
        Parameters
        ----------
        v: (n,) numpy array,
        n-dimensional vector to project
        s: int, optional, default: 1,
        radius of the simplex
        Returns
        -------
        w: (n,) numpy array,
        Euclidean projection of v on the simplex
        Notes
        -----
        The complexity of this algorithm is in O(n log(n)) as it involves sorting v.
        Better alternatives exist for high-dimensional sparse vectors (cf. [1])
        However, this implementation still easily scales to millions of dimensions.
        References
        ----------
        [1] Efficient Projections onto the .1-Ball for Learning in High Dimensions
        John Duchi, Shai Shalev-Shwartz, Yoram Singer, and Tushar Chandra.
        International Conference on Machine Learning (ICML 2008)
        http://www.cs.berkeley.edu/~jduchi/projects/DuchiSiShCh08.pdf
        """
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    n, = v.shape  # will raise ValueError if v is not 1-D
    # check if we are already on the simplex
    if v.sum() == s and np.alltrue(v >= 0):
        # best projection: itself!
        return v
    # get the array of cumulative sums of a sorted (decreasing) copy of v
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    # get the number of > 0 components of the optimal solution
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - s))[0][-1]
    # compute the Lagrange multiplier associated to the simplex constraint
    theta = (cssv[rho] - s) / (rho + 1.0)
    # compute the projection by thresholding v using theta
    w = (v - theta).clip(min=0)
    return w


def plot_results(primal, dual):
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(primal, label='Primal energy')
    plt.plot(dual, label='Dual energy')
    plt.legend()


def optimize_primal_dual(f, g, max_iter=1000, theta_ini=None, lmbd=100):
    nb_classes, h, w = f.shape
    tau_dual = 0.5
    tau_primal = 0.25
    smooth_factor = 0.7

    ####################
    ## INITIALIZATION ##
    ####################

    # Initialization of theta
    if theta_ini is None:
        theta_ini = np.random.rand(*f.shape)

    theta = theta_ini
    theta_tilde = np.copy(theta)
    projected_theta = np.zeros(f.shape)
    hist_theta = np.zeros((max_iter, *f.shape))

    # Dual variables and divergence results
    div = np.zeros(f.shape)

    xi = np.zeros((*f.shape, 2))
    for k in range(nb_classes):
        xi[k] = forward_gradient(theta_tilde[k])

    # Energies
    dual = []
    primal = []
    gap = []

    # Paths
    script_dir = os.path.dirname(__file__)
    results_dir = os.path.join(script_dir, 'output/')

    ##############################
    # PRIMAL - DUAL ITERATIONS
    ##############################

    it = 0
    stop = False

    while not stop and it < max_iter:

        theta_old = theta

        ##########################
        # DUAL UPDATE - ASCENT
        ##########################

        for k in range(nb_classes):
            xi[k] = xi[k] + tau_dual * forward_gradient(theta_tilde[k])

            ######################
            # PROJECTION ON Kg
            ######################

            norm_xi_k = np.linalg.norm(xi[k], ord=2, axis=2)
            mask = np.where(np.abs(norm_xi_k) > g / 2)
            xi[k][mask] = double(g)[mask] / 2 * xi[k][mask] / double(norm_xi_k)[mask]

        #############################
        # PRIMAL UPDATE - DESCENT
        #############################

        for k in range(nb_classes):
            div[k] = backward_divergence(xi[k])
            assert (np.count_nonzero(div[k]) > 0)  # Safety check

        theta = theta + tau_primal * (div - (1 / lmbd) * f)

        ###############################
        # PROJECTION ON THE SIMPLEX
        ###############################

        for i in range(h):
            for j in range(w):
                projected_theta[:, i, j] = euclidean_proj_simplex(theta[:, i, j])

        theta = projected_theta

        #########################
        ## OVERRELAXATION STEP ##
        #########################

        theta_tilde = theta + smooth_factor * (theta - theta_old)

        ######################
        ## COMPUTE ENERGIES ##
        ######################

        dual.append(compute_dual_energy(f, div, lmbd))
        primal.append(compute_primal_energy(theta_tilde, f, g, lmbd))
        gap.append(primal[-1] - dual[-1])

        print('Dual energy = {} ; Primal Energy = {} ; Gap = {} ; Iteration = {}'.format(dual[-1], primal[-1], gap[-1],
                                                                                         it))

        # Store theta in history
        hist_theta[it] = theta

        # TERMINATION CRITERIA

        stop = (it > 10) and (np.abs(dual[-1] - dual[-2]) < 0.001 * np.abs(dual[-1]))

        filepath = results_dir + 'state_' + str(it) + '.png'
        plt.imsave(filepath, np.argmax(hist_theta[it], axis=0) * 255, cmap='gray')

        it += 1
    hist_theta = hist_theta[:it]
    print('>> Saving gif...')
    save_gif(hist_theta)
    print('... Done')

    return primal, dual, hist_theta



def dice(ground_truth, best):
    """
    returns the dice metric only for two objects
    objects needs to be set to 0
    background sets to 1
    """

    XuY =  np.sum((ground_truth+best==0))
    X = np.sum(ground_truth==0)
    Y = np.sum(best==0)

    dice = np.round(100*2*XuY/(X+Y), 2)

    return dice


