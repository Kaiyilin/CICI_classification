import scipy
import numpy as np
import tensorflow as tf

# one axes random rotation
def tf_random_rotate_image(image, label):
    def random_rotate_image(image):
        image = scipy.ndimage.rotate(image, np.random.uniform(-80, 80), order=0, reshape=False)
        return image
    im_shape = image.shape
    [image,] = tf.py_function(random_rotate_image, [image], [tf.float64])
    image.set_shape(im_shape)
    return image, label
