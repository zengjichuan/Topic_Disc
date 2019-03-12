from __future__ import print_function

import engine
import utils
from models.conv_models import INFER
import logging
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import pickle
from engine import LossManager
import torch
import itertools
import json

logger = logging.getLogger()


def generate(model, data_feed, data_seq, config, num_batch=1, dest_f=None):
    """
    Generate latent representation and visualization data
    :param model:
    :param data_feed:
    :param config:
    :param num_batch:
    :param dest_f:
    :return:
    """
    model.eval()
    old_batch_size = config.batch_size

    # if num_batch != None:
    #     config.batch_size = 5

    data_feed.epoch_init(config, ignore_residual=False, shuffle=False, verbose=False)
    config.batch_size = old_batch_size

    data_seq, msg_cnt, word_cnt = data_seq
    data_seq = list(itertools.chain.from_iterable(data_seq))  # flatten seq data

    logger.info("Generation: {} batches".format(data_feed.num_batch
                                                if num_batch is None
                                                else num_batch))
    gen_items = []

    weight_matrix = model.decoder.weight.data.cpu().numpy()       # vocab_size * (disc_num + topic_num)

    while True:
        batch = data_feed.next_batch()
        if batch is None or (num_batch is not None
                             and data_feed.ptr > num_batch):
            break
        rst = model(batch, mode=INFER, return_latent=True)
        latent = rst.gen.cpu().data.numpy()
        tar = batch['targets']
        # print(tar.shape)

        index_base = (data_feed.ptr - 1) * config.batch_size

        for b_id in range(tar.shape[0]):
            vocab_weight = []
            vocab_source = []
            for v_id in range(weight_matrix.shape[0]):
                att = latent[b_id] * weight_matrix[v_id]
                max_ind = np.argmax(att)
                max_val = att[max_ind]
                vocab_weight.append(max_val)
                vocab_source.append(max_ind)
            # filter with tar
            tar_ind = (tar[b_id]>0)
            vocab_weight = np.array(vocab_weight)[tar_ind]
            vocab_source = np.array(vocab_source)[tar_ind]

            # get target sent
            print(np.argwhere(tar_ind).ravel())

            # tar_str = engine.get_sent(model, np.argwhere(tar_ind).squeeze())
            tar_str_seq = engine.get_sent(model, data_seq[index_base + b_id].utt)

            # alias weight and source
            alias_index = map(list(np.argwhere(tar_ind).ravel()).index, data_seq[index_base + b_id].utt)
            vocab_weight = vocab_weight[alias_index]
            vocab_source = vocab_source[alias_index]

            logger.info("Target: {}".format(tar_str_seq))
            logger.info("Source: {}".format(vocab_source))
            logger.info("Weight: {}\n".format(vocab_weight))

            if dest_f is not None:
                gen_items.append({"target": tar_str_seq, "source": list(vocab_source.astype(np.int)), "weight": list(vocab_weight.astype(np.float))})

    if gen_items and dest_f is not None:
        json.dump(gen_items, dest_f, indent=4)
    logger.info("Generation Done")
