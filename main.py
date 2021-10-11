import os
import scipy
import datetime
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.keras.backend import mean
from dataloader.dataloader import ds
from configs import pjt_config
from c_models.resnet3dse_swish import ResnetSE3DBuilder_swish
from basicTools.classVisualisation import IntegratedGradients


# one axes random rotation
def tf_random_rotate_image(image, label):
    def random_rotate_image(image):
        image = scipy.ndimage.rotate(image, np.random.uniform(-80, 80), order=0, reshape=False)
        return image
    im_shape = image.shape
    [image,] = tf.py_function(random_rotate_image, [image], [tf.float64])
    image.set_shape(im_shape)
    return image, label


def main():
    parser = argparse.ArgumentParser()
    # Add '--image_folder' argument using add_argument() including a help. The type is string (by default):
    parser.add_argument('--log_output_path', type=str, default=None)
    parser.add_argument('--checkpoint_folder', type=str, default=None)
    parser.add_argument('--visualisation', type=bool, default=False)
    parser.add_argument('--opt', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batchsize', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--num_classes', type=int, default=3)

    # Parse the argument and store it in a dictionary:
    args = vars(parser.parse_args())
    print(args)


    # tensorboard log directiory
    logdir = args["log_output_path"] + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # model checkpoint
    checkpoint_dir = args["checkpoint_folder"] + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    checkpoint_prefix = os.path.join(checkpoint_dir, "weights-{epoch:02d}.hdf5")
    
    try:
        os.makedirs(logdir)
        os.makedirs(checkpoint_dir)
    except FileExistsError as e:
        print(e) 

    if args["opt"] == 1:
        optimiser = tf.keras.optimizers.Adam(learning_rate=args["lr"])
    elif args["opt"] == 2:
        optimiser = tf.keras.optimizers.RMSprop(learning_rate=args["lr"])
    else:
        optimiser = tf.keras.optimizers.SGD(learning_rate=args["lr"])

    strategy = tf.distribute.MirroredStrategy()
    print('\nNumber of GPU devices: {}'.format(strategy.num_replicas_in_sync))

    with strategy.scope():

        model = ResnetSE3DBuilder_swish.build_resnet_50(
            input_shape=pjt_config["model"]["input_fmri"], 
            num_outputs=args["num_classes"]
            )

        model.compile(
            optimizer=optimiser, 
            loss=pjt_config["train"]["loss"], 
            metrics=pjt_config["train"]["loss"]
            )
        print(model.summary())
                    
        callbacks = [ 
                    tf.keras.callbacks.TensorBoard(
                        log_dir=logdir, 
                        histogram_freq=1, 
                        write_graph=True, 
                        write_images=False,
                        update_freq='epoch', 
                        profile_batch=2, 
                        embeddings_freq=0,
                        embeddings_metadata=None
                        ),

                    tf.keras.callbacks.ModelCheckpoint(
                        filepath=checkpoint_prefix,
                        verbose=0,
                        save_weights_only=True,
                        save_freq='epoch'
                        )
                    ]

        model.fit(
            ds.map(tf_random_rotate_image)
            .shuffle(50)
            .batch(args["batchsize"]), 
            epochs=args["epochs"],
            callbacks=callbacks
            )

    if args["visualisation"]:
        ig_result = IntegratedGradients(model=model, dataset=ds, target_index=1)
        ig_result = ig_result.calculate_IG(img_shape=pjt_config["model"]["input_fmri"], m_steps=50)
        mean_ig_result = np.arange(ig_result, axis=0)
        for i in range(mean_ig_result.shape[3]):
            plt.imshow(mean_ig_result[:, :, i])
            plt.savefig(f"vis_{i+1}.png")
            plt.axis("off")
            plt.close()
    
    else:
        pass
    


if __name__ == "__main__":
    main()