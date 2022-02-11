import cv2, numpy as np
from sklearn.cluster import KMeans
from imageLoader import WoodType, read_sample_wood_pic
from skimage import io, color

def rgb_to_lab(rgb):
    return cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2LAB)[0][0]
    # convert since cv2.cvtColor has weird offset shit that happens
    # return np.array([lab[0] / 2.55, lab[1] - 128, lab[2] - 128])

def get_color_freqs(image, num_colors, use_lab_values=False):
    reshape = image.reshape((image.shape[0] * image.shape[1], 3))
    cluster = KMeans(n_clusters=num_colors).fit(reshape)
    return get_color_hist(cluster, cluster.cluster_centers_, use_lab_values=use_lab_values)


def get_color_hist(cluster, centroids, use_lab_values=False):
    labels = np.arange(0, len(np.unique(cluster.labels_)) + 1)
    (hist, _) = np.histogram(cluster.labels_, bins = labels)
    hist = hist.astype("float")
    hist /= hist.sum()

    srt = sorted([(percent, color) for (percent, color) in zip(hist, centroids)])
    if not use_lab_values:
        return srt
    else:
        for i in range(0, len(srt)):
            srt[i] = tuple([srt[i][0], rgb_to_lab(srt[i][1])])
        return srt

def visualize_colors(cluster, centroids):
    # Get the number of different clusters, create histogram, and normalize
    colors = get_color_hist(cluster, centroids)

    # Create frequency rect and iterate through each cluster's color and percentage
    rect = np.zeros((50, 300, 3), dtype=np.uint8)
    start = 0
    for (percent, color) in colors:
        print(color, "{:0.2f}%".format(percent * 100))
        end = start + (percent * 300)
        cv2.rectangle(rect, (int(start), 0), (int(end), 50), \
                      color.astype("uint8").tolist(), -1)
        start = end
    return rect


# don't think this is useful but I don't feel like checking right now
# TODO: probably delete this.
def run():
    # Load image and convert to a list of pixels
    image = read_sample_wood_pic(WoodType.ALDER)
    reshape = image.reshape((image.shape[0] * image.shape[1], 3))

    # Find and display most dominant colors
    cluster = KMeans(n_clusters=5).fit(reshape)
    visualize = visualize_colors(cluster, cluster.cluster_centers_)
    visualize = cv2.cvtColor(visualize, cv2.COLOR_RGB2BGR)
    # cv2.imshow('visualize', visualize)
    cv2.waitKey()