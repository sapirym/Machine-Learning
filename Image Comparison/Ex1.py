import numpy as np
import matplotlib.pyplot as plt
from init_centroids import init_centroids
from load import load


def k_means_algo(pixels, k):
    """
    K means algorithm - calculate centroids.
    :param pixels: pixels array
    :param k: number of means
    :return: loss map for each centroid
    """
    centroids_initialize = init_centroids(pixels, k);
    print('k=' + str(k) + ':');
    loss_map = [];

    # run 11 epochs
    for iter in range(11):
        print(print_centroids(centroids_initialize, iter));
        dict_centroids = {};
        loss = 0;

        # run on each pixel
        for pixle in pixels:
            minimum = float('inf');
            for index in range(k):

                # calculate the euclidean square distance between the pixel to the centroid
                min_dist = pow(np.linalg.norm(centroids_initialize[index] - pixle), 2);

                # choose the minimal distance
                if min_dist < minimum:
                    minimum = min_dist;
                    index_min = index;

            loss += minimum;
            # add to dictionary the pixel with minimum distance with key= index of centroid
            try:
                dict_centroids[index_min].append(pixle);
            except KeyError:
                dict_centroids[index_min] = [pixle]

        # calculate the average of each centroid
        for key in dict_centroids.keys():
            centroids_initialize[key] = np.average(dict_centroids[key], axis=0);
        loss_avg = loss / float(len(pixels));
        loss_map.append(loss_avg);

    return loss_map;

def print_centroids(cent, iteration_number):
    """
    return string that contain the centroid in specific iteration.
    :param cent: the current centroids values
    :param iteration_number: iteration number
    :return: string to print
    """
    print('iter ' + str(iteration_number) + ': ');
    if type(cent) == list:
        cent = np.asarray(cent)
    if len(cent.shape) == 1:
        return ' '.join(str(np.floor(100*cent)/100).split()).replace('[ ', '[').replace('\n', ' ').replace(' ]', ']').replace(' ', ', ')
    else:
        return ' '.join(str(np.floor(100*cent)/100).split()).replace('[ ', '[').replace('\n', ' ').replace(' ]', ']').replace(' ', ', ')[1:-1]


def plot_loss(loss, k):
    """
    plot loss on graph
    :param loss:
    :param k:
    :return:
    """
    y = list(range(11))
    plt.plot(y, loss);
    plt.ylabel('Loss')
    plt.xlabel('Iteration')
    plt.title('The loss with K=' + str(k));
    plt.show()


if __name__ == '__main__':
    """
    Main function
    """
    data, img_size = load();
    for k in [2, 4, 8, 16]:
        loss_map = k_means_algo(data, k);
