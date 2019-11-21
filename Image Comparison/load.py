from scipy.misc import imread


def load():
    """
    Load image and normalize it .
    :return: normalized pixels and image size.
    """
    path = 'dog.jpeg'
    A = imread(path)
    A_norm = A.astype(float) / 255.
    img_size = A_norm.shape
    X = A_norm.reshape(img_size[0] * img_size[1], img_size[2])
    return X, img_size;