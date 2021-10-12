import tensorflow as tf

pjt_config = {
   "database" : {
        "auth" : "",
        "db_name" : "",
        "collection_name" : ""
   },
   "train" : {
       "loss" : tf.keras.losses.CategoricalCrossentropy(),
       "metrics": tf.keras.metrics.CategoricalAccuracy()
   },
   "model": {
       "input_fmri": (64, 64, 64, 1),
       "input_diffusion": (128, 128, 128, 1)
   },
   "path" : {
        "logs" : "./logs/",
        "ckpt_dir" : "./checkpoint_folder/"
   }
}
