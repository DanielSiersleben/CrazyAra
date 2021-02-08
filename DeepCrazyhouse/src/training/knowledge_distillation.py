import glob
import os
import shutil

import mxnet as mx
# import mxnet.contrib.onnx as onnx_mxnet
import numpy as np
import zarr
from numcodecs import Blosc

import matplotlib.pyplot as plt

from DeepCrazyhouse.src.domain.util import normalize_input_planes
from DeepCrazyhouse.src.domain.variants.constants import NB_CHANNELS_TOTAL, BOARD_HEIGHT, BOARD_WIDTH
from DeepCrazyhouse.src.preprocessing.dataset_loader import load_pgn_dataset
from DeepCrazyhouse.src.training.trainer_agent_mxnet import fill_up_batch, prepare_policy


def augment_training_set_for_kd(filepath_to_dataset, filepath_to_symbol, filepath_to_params, batch_size,
                                filepath_to_kd_data=None,
                                model_abbreviation=None):
    """
    Creates a Dataset for Knowledge Distillation using Value and Policy output from the given
    ONNX-Model (path_to_model) as new targets for the newly created Dataset at kd_data_directory
    """

    ## load first chunk of training set to assign dimensions
    _, x_train, yv_train, yp_train, _, _ = load_pgn_dataset(dataset_type="train",
                                                            part_id=0,
                                                            normalize=True,
                                                            verbose=False,
                                                            q_value_ratio=0)

    input_shape = x_train[0].shape
    yp_train = prepare_policy(yp_train, True, False)

    train_iter = mx.io.NDArrayIter({'data': x_train},
                                         {'value_label': yv_train, 'policy_label': yp_train},
                                         batch_size,
                                         shuffle=True)

    # set label names
    value_name = "y_value_prediction"
    policy_name = "y_policy_prediction"
    if model_abbreviation is not None:
        value_name += "_" + model_abbreviation
        policy_name += "_" + model_abbreviation

    ##load the model
    symbol = mx.sym.load(filepath_to_symbol)
    model = mx.mod.Module(symbol=symbol, context=mx.gpu(), label_names=['value_label', 'policy_label'])
    model.bind(for_training=False, data_shapes=[('data', (batch_size, input_shape[0], input_shape[1], input_shape[2]))],
               label_shapes=train_iter.provide_label)
    model.load_params(filepath_to_params)

    if filepath_to_kd_data is None:
        zarr_filepaths = glob.glob(filepath_to_dataset + "**/*.zip")
    else:
        assert not os.path.exists(filepath_to_kd_data), "new directory has to be empty"
        ## first copy dataset to destination
        shutil.copytree(filepath_to_dataset, filepath_to_kd_data)
        zarr_filepaths = glob.glob(filepath_to_kd_data + "**/*.zip")

    ##loop over dataset
    for zarr_file in zarr_filepaths:
        print("augmenting:" + zarr_file)

        store = zarr.ZipStore(zarr_file, mode='a')
        data = zarr.group(store=store)

        x = np.array(data["x"])

        # normalize Data
        x = x.astype(np.float32)
        x *= normalize_input_planes(np.ones((NB_CHANNELS_TOTAL, BOARD_HEIGHT, BOARD_WIDTH)))

        y_policy = np.ndarray(shape=(len(x), yp_train.shape[1]), dtype=np.float16)
        y_value = np.ndarray(shape=[len(x)])

        if len(x) < batch_size:
            x = fill_up_batch(x, batch_size)

        train_iter = mx.io.NDArrayIter(data=x, batch_size=batch_size)

        iteration = 0
        ## compute policy and value for all data
        for batch in train_iter:
            model.forward(batch)

            indices = np.arange(start=iteration * batch_size, stop=(iteration + 1) * batch_size, step=1)

            value, policy = model.get_outputs()
            policy = policy.asnumpy()
            value = value.asnumpy()
            policy[abs(policy) < 1e-3] = 0

            if indices[-1] >= len(x):  # last iteration of file
                indices = np.arange(start=iteration * batch_size, stop=len(x), step=1)
                indices_local = np.arange(0, len(indices))

                y_value[indices] = value[indices_local, 0]
                y_policy[indices, :] = policy[indices_local, :]

            else:
                y_value[indices] = value[:, 0]
                y_policy[indices, :] = policy

            iteration += 1

        # define the compressor object
        compressor = Blosc(cname="lz4", clevel=5, shuffle=Blosc.SHUFFLE)
        # create the label arrays and copy the labels data in them
        data.create_dataset(
            name=value_name, shape=y_value.shape, dtype=np.float16, data=y_value,
            synchronizer=zarr.ThreadSynchronizer(), overwrite=True
        )
        data.create_dataset(
            name=policy_name,
            shape=y_policy.shape,
            dtype=np.float16,
            data=y_policy,
            chunks=(128, y_policy.shape[1]),
            synchronizer=zarr.ThreadSynchronizer(),
            compression=compressor,
        )
        store.close()


def clip_quantile(distribution, q):
    quantile_val = np.quantile(distribution, q)
    distribution[abs(distribution) < q] = 0
    return distribution


augment_training_set_for_kd(
    #filepath_to_dataset=r"C:\Users\Daniel\Desktop\UNI\BSC Thesis\Train_data\planes\train\2018-09-27-10-43-39",
    filepath_to_dataset=r"E:\KD_DATA_old",
    filepath_to_symbol=r"C:\Users\Daniel\Desktop\UNI\BSC Thesis\nn\risev2_27_blocks_crazyhouse\model\model-1.17985-0.606-symbol.json",
    filepath_to_params=r"C:\Users\Daniel\Desktop\UNI\BSC Thesis\nn\risev2_27_blocks_crazyhouse\model\model-1.17985-0.606-0212.params",
    batch_size=1024,
    model_abbreviation="risev2_27")
