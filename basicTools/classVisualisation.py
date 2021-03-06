import tensorflow as tf
import sys

from tensorflow.python.keras.backend import ndim

# for visualisation, a model must be contained when calling Visualisation methods
class Visualisation(object):

    def __init__(self, model: tf.keras.models, dataset, target_index) -> None:
        super().__init__()
        self.model = model
        self.dataset = dataset
        self.target_index = target_index

class IntegratedGradients(Visualisation):
    """
    The IntegratedGradients algorithm is an Explainable AI technique 
    introduced in the paper Axiomatic Attribution for Deep Networks. 

    url: https://arxiv.org/abs/1703.01365
    
    IG aims to explain the relationship between a model's predictions in terms of its features. 
    It has many use cases including understanding feature importances, 
    identifying data skew, and debugging model performance.

    This algorithm can be used on traditional NN 

    but the machine learnig model required further check
    """  

    @staticmethod
    def interpolate_images(baseline, image, alphas):

        if image.dtype != 'float32':
            image = tf.convert_to_tensor(image)
            image = tf.dtypes.cast(image, dtype=tf.float32)
        else:
            image = tf.convert_to_tensor(image)
        img_ndim = image.ndim
        #print(img_ndim)
        if img_ndim == 3:
            alphas_x = alphas[:, tf.newaxis, tf.newaxis, tf.newaxis]
        elif img_ndim ==1:
            alphas_x = alphas[:, tf.newaxis]
        elif img_ndim == 4:    
            alphas_x = alphas[:, tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis]
        else:
            sys.exit("dimension not fit, check your image dimension")
        baseline_x = tf.expand_dims(baseline, axis=0)
        input_x = tf.expand_dims(image, axis=0)
        delta = input_x - baseline_x
        images = baseline_x +  alphas_x * delta

        return images
    @staticmethod
    def compute_gradients(images, model, target_class_idx):
        with tf.GradientTape() as tape:
            tape.watch(images)
            logits = model(images)
            probs = tf.nn.softmax(logits, axis=-1)[:, target_class_idx]

        return tape.gradient(probs, images)
    @staticmethod
    def integral_approximation(gradients):
        # riemann_trapezoidal
        grads = (gradients[:-1] + gradients[1:]) / tf.constant(2.0)
        integrated_gradients = tf.math.reduce_mean(grads, axis=0)
        return integrated_gradients
    
    def calculate_IG(self, img_shape: tuple, m_steps: int):
        """
        This function will generate the integrated gradients for each image
        
        input: numpy array
        output: list of numpy array

        img_shape: Given an image shape, that fits your model to generate a black tensor
        m_steps: number of steps toward your target, in other words your own image 

        """
        assert type(m_steps) == int , "the m_steps should be an integer"
        assert type(img_shape) == tuple, "the image shape should be a tuple"
        baseline = tf.zeros(shape=img_shape, dtype='float32')
        # Generate m_steps intervals for integral_approximation() below.
        alphas = tf.linspace(start=0.0, stop=1, num=m_steps+1) 
        ig_results = [] 
        for i in range(len(self.dataset)):
            target_img = self.dataset[i]
            interpolated_images = IntegratedGradients.interpolate_images(baseline=baseline, 
                                                                         image=target_img, 
                                                                         alphas=alphas)

            path_gradients = IntegratedGradients.compute_gradients(images=interpolated_images, 
                                                                   model=self.model, 
                                                                   target_class_idx=self.target_index)
            #print(path_gradients.shape)
            ig = IntegratedGradients.integral_approximation(gradients=path_gradients)
            # convert to numpy array
            ig = ig.numpy()
            ig_results.append(ig)
            print(f"{(i+1)/len(self.dataset)*100}% completed")
        return ig_results