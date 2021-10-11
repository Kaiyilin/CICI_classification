import os
import pymongo
import tensorflow as tf
from basicTools.classVisualisation import IntegratedGradients
from models.resnet3d_variation import ResnetSE_3DBuilder
from sklearn.metrics import ( 
    roc_curve, 
    roc_auc_score, 
    precision_score, 
    recall_score, 
    accuracy_score, 
    f1_score
    )

## hyper parameters
_lr = 1e-3
_mongo_auth = ""
_opt_sgd = tf.keras.optimizers.SGD(lr= _lr, momentum=0.9, nesterov=True)
_loss_func = tf.keras.losses.CategoricalCrossentropy()
_buffer_size = 120
_batch_size = 5
_num_classes = 3
db_name = ""
collection_name = ""


strategy = tf.distribute.MirroredStrategy()
print('\nNumber of GPU devices: {}'.format(strategy.num_replicas_in_sync))

with strategy.scope():
    os.chdir(logdir)
    model = ResnetSE_3DBuilder.build_resnet_50(input_shape = imgs[0].shape, num_outputs = _num_classes)
    print("\nBuilding a Res_Net50 model")
    model.compile(optimizer=  _opt_sgd, loss= _loss_func)
    print(model.summary())
                
    callbacks = [ 
                tf.keras.callbacks.TensorBoard(log_dir=logdir, 
                                                    histogram_freq=1, 
                                                    write_graph=True, 
                                                    write_images=False,
                                                    update_freq='epoch', 
                                                    profile_batch=2, 
                                                    embeddings_freq=0,
                                                    embeddings_metadata=None),
                tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix,
                                                   verbose=0,
                                                   save_weights_only=True,
                                                   save_freq='epoch')
                ]

    history = model.fit()

ig_result = IntegratedGradients(model = model, dataset = imgs, target_index = 1)
ig_result = ig_result.calculate_IG(img_shape = imgs[0].shape, m_steps = 50)


pred_prob = model.predict("testImages")
pred_trans = np.argmax(pred_prob, axis = 1)
fpr, tpr, thresh = roc_curve("test_labels", pred_prob[:, 1])
auc_score = roc_auc_score("test_labels", pred_prob[:, 1])
accu = accuracy_score("test_labels", pred_trans)
recall = recall_score("test_labels", pred_trans)
precision = precision_score("test_labels", pred_trans)
f1 = f1_score("test_labels", pred_trans)
plt.title(f'ResNet50_{i}_ROC')
plt.plot(fpr, tpr, linestyle = '-', label = f"ResNet50_{i} auc({auc_score:.2f})")
plt.plot([0 , 1], [0, 1], linestyle = '--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive rate')
plt.legend(loc = 'lower right')    
plt.savefig(f"Res50_roc_{i}.png", dpi = 500)
plt.close()

