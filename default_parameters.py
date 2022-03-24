
###########################################################################################
# Baseline:
# - alpha=30
# - sigma=30
# - LMBD=100
# - NB_ITERS = 100
###########################################################################################

showScribble = False
showDistanceMap = False
showSegmentation = True
showGroundTruth = True
NB_ITERS = 20
alpha = 30  # Distance parameter: indicates how segmentation is locally sensible to the input scribble
sigma = 30  # Smoothing parameter: indicates how segmentation depends on color
LMBD = 100  # Regularization parameter:
eps = 10e-7
USE_DIST = True
USE_COLOR = True
MARGINAL = False

default_shape = (240, 320)
down_shape = (240 / 2, 320 / 2)

color_palette = [[0.88, 0.1, 0.29], [0.98, 0.64, 0.36], [0.99, 0.91, 0.60], [0.63, 0.85, 0.64], [0.28, 0.62, 0.70], [0.28, 0.32, 0.70], [0.90, 0.62, 0.70]]

