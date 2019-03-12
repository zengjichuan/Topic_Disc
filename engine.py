# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
from models.model_bases import summary
import torch
import os
import pickle
from collections import defaultdict
from models.conv_models import INFER, TRAIN
import logging
from sklearn.metrics import accuracy_score, f1_score
import json
logger = logging.getLogger()


class LossManager(object):
    def __init__(self):
        self.losses = defaultdict(list)
        self.backward_losses = []

    def add_loss(self, loss):
        for key, val in loss.items():
            if val is not None and type(val) is not bool:
                self.losses[key].append(val.data[0])

    def add_backward_loss(self, loss):
        self.backward_losses.append(loss.data[0])

    def clear(self):
        self.losses = defaultdict(list)
        self.backward_losses = []

    def pprint(self, name, window=None, prefix=None):
        str_losses = []
        for key, loss in self.losses.items():
            if loss is None:
                continue
            avg_loss = np.average(loss) if window is None else np.average(loss[-window:])
            str_losses.append("{} {:.3f}".format(key, avg_loss))
            if 'nll' in key:
                str_losses.append("PPL({}) {:.3f}".format(key, avg_loss))
        if prefix:
            return "{}: {} {}".format(prefix, name, " ".join(str_losses))
        else:
            return "{} {}".format(name, " ".join(str_losses))

    def avg_loss(self):
        return np.mean(self.backward_losses)


def print_topic_words(decoder, vocab_dic, n_top_words=10):
    beta_exp = decoder.weight.data.cpu().numpy().T
    for k, beta_k in enumerate(beta_exp):
        topic_words = [vocab_dic[w_id] for w_id in np.argsort(beta_k)[:-n_top_words-1:-1]]
        yield 'Topic {}: {}'.format(k, ' '.join(x.encode('utf-8') for x in topic_words))


def get_sent(model, data):
    sent = [model.vocab_bow[w_id] for w_id in data]
    return sent


def train(model, train_feed, valid_feed, test_feed, config):

    patience = 10  # wait for at least 10 epoch before stop
    valid_loss_threshold = np.inf
    best_valid_loss = np.inf
    batch_cnt = 0
    optimizer = model.get_optimizer(config)
    done_epoch = 0
    train_loss = LossManager()
    model.train()
    logger.info(summary(model, show_weights=False))
    logger.info("**** Training Begins ****")
    logger.info("**** Epoch 0/{} ****".format(config.max_epoch))

    while True:
        train_feed.epoch_init(config, verbose=done_epoch==0, shuffle=True)
        while True:
            batch = train_feed.next_batch()
            if batch is None:
                break

            optimizer.zero_grad()
            loss = model(batch)
            if model.flush_valid:
                logger.info("Flush previous valid loss")
                best_valid_loss = np.inf
                model.flush_valid = False
                optimizer = model.get_optimizer(config)

            model.backward(batch_cnt, loss)
            optimizer.step()
            batch_cnt += 1
            train_loss.add_loss(loss)

            if batch_cnt % config.print_step == 0:
                logger.info(train_loss.pprint("Train", window=config.print_step,
                                              prefix="{}/{}-({:.3f})".format(batch_cnt % config.ckpt_step,
                                                                         config.ckpt_step,
                                                                         model.kl_w)))
                # update l1 strength
                if config.use_l1_reg and batch_cnt <= config.freeze_step:
                    model.reg_l1_loss.update_l1_strength(model.ctx_decoder.weight)

            if batch_cnt % config.ckpt_step == 0:
                logger.info("\n=== Evaluating Model ===")
                logger.info(train_loss.pprint("Train"))
                done_epoch += 1

                # validation
                logging.info("Discourse Words:")
                logging.info('\n'.join(print_topic_words(model.x_decoder, model.vocab_bow)))
                logging.info("Topic Words:")
                logging.info("\n".join(print_topic_words(model.ctx_decoder, model.vocab_bow)))

                valid_loss = validate(model, valid_feed, config, batch_cnt)

                # update early stopping stats
                if valid_loss < best_valid_loss:
                    if valid_loss <= valid_loss_threshold * config.improve_threshold:
                        patience = max(patience,
                                       done_epoch * config.patient_increase)
                        valid_loss_threshold = valid_loss
                        logger.info("Update patience to {}".format(patience))

                    if config.save_model:
                        logger.info("Model Saved.")
                        torch.save(model.state_dict(),
                                   os.path.join(config.session_dir, "model"))

                    best_valid_loss = valid_loss

                if done_epoch >= config.max_epoch \
                        or config.early_stop and patience <= done_epoch:
                    if done_epoch < config.max_epoch:
                        logger.info("!!Early stop due to run out of patience!!")

                    logger.info("Best validation loss %f" % best_valid_loss)

                    return

                # exit eval model
                model.train()
                train_loss.clear()
                logger.info("\n**** Epcoch {}/{} ****".format(done_epoch,
                                                       config.max_epoch))


def validate(model, valid_feed, config, batch_cnt=None):
    model.eval()
    valid_feed.epoch_init(config, shuffle=False, verbose=True)
    losses = LossManager()
    while True:
        batch = valid_feed.next_batch()
        if batch is None:
            break
        loss = model(batch)
        losses.add_loss(loss)
        losses.add_backward_loss(model.model_sel_loss(loss, batch_cnt))

    valid_loss = losses.avg_loss()

    logger.info(losses.pprint(valid_feed.name))
    logger.info("Total valid loss {}".format(valid_loss))

    return valid_loss


def inference(model, data_feed, config, num_batch=1, dest_f=None):
    model.eval()

    data_feed.epoch_init(config, ignore_residual=False, shuffle=num_batch is not None, verbose=False)

    logger.info("Inference: {} batches".format(data_feed.num_batch
                                                if num_batch is None
                                                else num_batch))
    gen = []
    d_ids = []
    while True:
        batch = data_feed.next_batch()
        if batch is None or (num_batch is not None
                             and data_feed.ptr > num_batch):
            break
        outputs = model(batch, mode=INFER)

        # move from GPU to CPU
        gen_ = outputs.gen.cpu().data.numpy()
        d_ids_ = outputs.d_ids.cpu().data.numpy()

        gen.append(gen_)
        d_ids.append(d_ids_)
    gen = np.concatenate(gen)
    # output discourse
    d_ids = np.concatenate(d_ids)
    rst = []

    for r_id, row in enumerate(data_feed.data):
        u_id = row.target.meta["id"]
        disc = row.target.meta["disc"]
        vec = gen[r_id]
        d_id = d_ids[r_id][0]
        rst.append({"id": u_id, "true_disc": disc, "pred_disc": d_id, "vec": vec})

    pickle.dump(rst, dest_f)
    logger.info("Inference Done")

