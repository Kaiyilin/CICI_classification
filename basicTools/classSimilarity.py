import math
from decimal import Decimal
import numpy as np
 
class Similarity:

    def __init__(self, x, y):
        self.__x = x.flatten()
        self.__y = y.flatten()    

    def euclidean_distance(self):
 
        """ return euclidean distance between two lists """
 
        return math.sqrt(sum(pow(a - b, 2) for a, b in zip(self.__x, self.__y)))
    
    def manhattan_distance(self):
 
        """ return manhattan distance between two lists """
 
        return sum(abs(a - b) for a, b in zip(self.__x, self.__y))
    

    def cosine_similarity(self):
 
        """ return cosine similarity between two lists """
 
        numerator = sum(a * b for a, b in zip(self.__x, self.__y))
        denominator = math.sqrt(sum([a * a for a in self.__x]))* math.sqrt(sum([b * b for b in self.__y]))

        return round(numerator / float(denominator), 3)


    def jaccard_similarity(self):
 
        """ returns the jaccard similarity between two lists """
 
        intersection_cardinality = len(set.intersection([set(self.__x), set(self.__y)]))
        union_cardinality = len(set.union([set(self.__x), set(self.__y)]))

        return intersection_cardinality/float(union_cardinality)

    def psnr(self):
        mse = (np.abs(self.__x - self.__y) ** 2).mean()
        psnr = 10 * np.log10(self.__x.max() * self.__y.max() / mse)

        return psnr
    
    def ssim(self, max_val):
        import tensorflow as tf
        img1 = tf.convert_to_tensor(self.__x)
        img2 = tf.convert_to_tensor(self.__y)
        
        return tf.image.ssim(self.__x, self.__y, max_val, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03)
        