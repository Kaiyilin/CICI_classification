import pymongo
import numpy as np
import tensorflow as tf
from basicTools.classData import Data
from configs import pjt_config
 

client = pymongo.MongoClient(pjt_config["database"]["auth"])
mydb = client[pjt_config["database"]["db_name"]]
mycollection = mydb[pjt_config["database"]["mycollection"]]
docs = mycollection.find({},{"Images.mfalff":1, "Labels":1})  

imgs = []
labels = []
for doc in docs: 
    img = np.array(doc["Images"]["mfalff"]) 
    img = Data.normalised(image=img)
    labels.append(doc["Labels"])
    img = img[None, ...] 
    imgs.append(img) 
imgs = np.concatenate(imgs, axis = 0)

# create tf data dataset
ds = tf.data.Dataset.from_tensor_slices(imgs, labels)
