 # Multiple Inputs
import os
import sys
import datetime
import nibabel as nib
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
print('\nImport completed')


# Data Preprocessing
def data_preprocessing(image):
    image = (image - image.min())/(image.max() - image.min()) 
    return image

def standardised(image): 
    img = (image - image.mean())/image.std() 
    return img 

# Read files
def myreadfile(dirr):
    """
    This version can import 3D array regardless of the size
    """
    os.chdir(dirr)
    #cwd = os.getcwd()

    number = 0

    flag = True
    imgs_array = np.array([])
    path_list=[f for f in os.listdir(dirr) if not f.startswith('.')]
    path_list.sort() #對讀取的路徑進行排序
    for file in path_list:
          if file.endswith(".nii"):
            #print(os.path.join(dirr, file))
            img = nib.load(os.path.join(dirr, file))
            img_array = img.get_fdata()
            img_array = data_preprocessing(img_array)
            img_array = img_array.reshape(-1,img_array.shape[0],img_array.shape[1],img_array.shape[2])
            number += 1
            if flag == True:
                imgs_array = img_array

            else:
                imgs_array = np.concatenate((imgs_array, img_array), axis=0)

            flag = False
    return number, imgs_array, path_list

def padding_zeros(array, pad_size):
    # define padding size
    elements = array.shape    
    for element in elements:
        if element > pad_size:
            sys.exit('\nThe expanded dimension shall be greater than your current dimension')
    pad_list = list() 
    for i in range(array.ndim):
        x = pad_size - array.shape[i]
        if x%2 ==1:
            y_1 = (x/2 +0.5)
            y_2 = (x/2 -0.5)
            z = (int(y_1),int(y_2))
            pad_list.append(z)

        else:
            y = int(x/2)
            z=(y,y)
            pad_list.append(z)
    pad_array = np.pad(array, pad_list, 'constant')
    pad_list = list() 
    return pad_array

def myreadfile_pad(dirr, pad_size):
    os.chdir(dirr)
    number = 0

    flag = True
    imgs_array = np.array([])
    path_list=[f for f in os.listdir(dirr) if not f.startswith('.')]
    path_list.sort()
    for file in path_list:
        if file.endswith(".nii"):
            #print(os.path.join(dirr, file))
            img = nib.load(os.path.join(dirr, file))
            img_array = img.get_fdata()
            img_array = data_preprocessing(img_array)
            img_array = padding_zeros(img_array, pad_size)
            img_array = img_array.reshape(-1,img_array.shape[0],img_array.shape[1],img_array.shape[2])
            number += 1
            if flag == True:
                imgs_array = img_array

            else:
                imgs_array = np.concatenate((imgs_array, img_array), axis=0)

            flag = False
    return number, imgs_array, path_list


def split(c,array):
    array_val = array[:c,:,:,:]
    array_tr = array[c:,:,:,:]
    return array_tr, array_val

def base_model_creator(model, train_para = False):
    base_model = tf.keras.models.Model([model.inputs], [model.get_layer(index=-2).output])
    base_model.trainable = train_para
    return base_model

def model_structure(model):
    """
    Visualise model's architecture
    display feature map shapes
    """
    for i in range(len(model.layers)):
      layer = model.layers[i]
    # summarize output shape
      print(i, layer.name, layer.output.shape)

def decay(epoch):
  if epoch <= 30:
    return 1e-1
  elif epoch > 30 and epoch <= 70:
    return 1e-2
  else:
    return 1e-3

# Setting callbacks

# tensorboard log directiory
logdir="/home/kaiyi/logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# model checkpoint
checkpoint_dir ="/home/kaiyi/trckpt/"+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
checkpoint_prefix = os.path.join(checkpoint_dir, "weights-{epoch:02d}.hdf5")
class PrintLR(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    print('\nLearning rate for epoch {} is {}'.format(epoch + 1,
                                                      model.optimizer.lr.numpy()))



print(checkpoint_dir)
print(logdir)

print("\ntf.__version__ is", tf.__version__)
print("\ntf.keras.__version__ is:", tf.keras.__version__)