import os
import glob
import numpy as np
import tensorflow as tf
import skimage.io as io

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

from models.unet import UNet

os.environ["CUDA_VISIBLE_DEVICES"]=""


# Read Annotation files
def find_file_path(dpath, slide):
    file_path = 'None'
    for f in dpath:
        file_name = f.split('/')[-1].split('.')[0]
        if file_name == slide:
            file_path = f
    return file_path


if __name__ == '__main__':
    # Types of annotation
    tissueTypeList = ['outside_roi', 'tumor', 'stroma', 'lymphocytic_infiltrate', 'necrosis_or_debris',
                      'glandular_secretions', 'blood', 'exclude', 'metaplasia_NOS', 'fat', 'plasma_cells',
                      'other_immune_infiltrate', 'mucoid_material', 'normal_acinus_or_duct', 'lymphatics',
                      'undetermined', 'nerve', 'skin_adnexa', 'blood_vessel', 'angioinvasion', 'dcis', 'other']

    # Select tissue type
    tissueType = 3
    sampleIndex = 0
    xTrainList = []
    yTrainList = []
    input_size = 256
    batch_size = 16
    isdata = True

    if isdata:
        global_X_train = np.load('/Users/leesan/datasets/BCSS/global_X_train_lymphocytic_infiltrate_20x.npy')
        global_Y_train = np.load('/Users/leesan/datasets/BCSS/global_Y_train_lymphocytic_infiltrate_20x.npy')
    else:
        # Location of annotated images
        data_dir = '/home/leesan/workspace/CrowdsourcingDataset-Amgadetal2019/sliced/'
        imgPaths = glob.glob(data_dir + 'images/*.png')
        annotPaths = glob.glob(data_dir + 'annots/*.png')

        # Collect training data
        for imgPath in imgPaths:
            f_name = imgPath.split('/')[-1].split('.')[0]
            annotPath = find_file_path(annotPaths, f_name)
            img = io.imread(imgPath)
            annot = io.imread(annotPath)
            if len(annot[annot == 0]) == 0:
                xTrainList.append(img)
                yTrainList.append(annot == tissueType)
                sampleIndex += 1

        global_X_train = np.reshape(xTrainList, (sampleIndex, input_size, input_size, 3))
        global_Y_train = np.reshape(yTrainList, (sampleIndex, input_size, input_size, 1)).astype(np.bool)

    # Split the data in training and remaining dataset
    X_train, X_test, y_train, y_test = train_test_split(global_X_train, global_Y_train, train_size=0.8)

    # Set hyper parameters
    input_shape = (input_size, input_size, 3)
    patch_shape = (input_size, input_size)
    batch_size = 16
    epochs = 50

    metrics = [
        tf.keras.metrics.AUC(curve='ROC', name='AuROC'),
    ]

    # Select model
    arch = UNet(input_shape,
                out_channels=1,
                nr_feats=64,
                nr_blocks=4,
                nr_conv=2,
                upscale="nn",
                drop_rate=0,
                last_activation="sigmoid",
                normalization="batch",
                nr_groups=-1,
                conv_type="full",
                name="Unet",
                initializer="truncated_normal",
                weight_decay=1e-5,
                weight_decay_type="l1")

    # Validation 5 folds
    kf = KFold(n_splits=5, shuffle=True, random_state=4)
    k_fold = kf.split(X_train, y_train)
    list_auc = []
    tf.debugging.set_log_device_placement(True)
    gpus = tf.config.list_logical_devices('GPU')
    strategy = tf.distribute.MirroredStrategy(gpus)
    with strategy.scope():
        for k, (train_idx, val_idx) in enumerate(k_fold):
            print(k)
            model = arch.net()
            model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(), weighted_metrics=metrics)
            model.summary()
            model.fit(X_train[train_idx, :], y_train[train_idx], epochs=epochs)

    # Save model
    # model.save(str(tissueTypeList[tissueType]) + '_256_256_20x')
    # model.save(str(tissueTypeList[tissueType]) + '_256_256_20x.h5')

