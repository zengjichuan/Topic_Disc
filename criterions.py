# -*- coding: utf-8 -*-
from __future__ import print_function
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from torch.autograd import Variable
import numpy as np
import torch
from utils import INT, FLOAT, LONG, cast_type
import logging


class L2Loss(_Loss):

    logger = logging.getLogger()
    def forward(self, state_a, state_b):
        if type(state_a) is tuple:
            losses = 0.0
            for s_a, s_b in zip(state_a, state_b):
                losses += torch.pow(s_a-s_b, 2)
        else:
            losses = torch.pow(state_a-state_b, 2)
        return torch.mean(losses)


class L1RegLoss(torch.nn.L1Loss):
    """
    get l1 loss according to target sparsity
    """
    logger = logging.getLogger()
    def __init__(self, sparsity):
        super(L1RegLoss, self).__init__()
        self.sparsity = sparsity
        self.l1_strength = 0.001

    def forward(self, input, target):
        l1_loss = super(L1RegLoss, self).forward(input, target)
        return self.l1_strength * l1_loss

    def update_l1_strength(self, input):
        # check sparsity
        num_zeros = (input.abs() < 1e-3).sum().float()
        cur_sparsity = num_zeros / (input.shape[0] * input.shape[1])
        diff = self.sparsity - cur_sparsity
        self.l1_strength *= sum(2.0 ** diff)
        self.logger.info("Current sparsity: %.3f, Update l1 strength to %.3f" % (cur_sparsity, self.l1_strength))

class NLLEntropy(_Loss):

    logger = logging.getLogger()
    def __init__(self, padding_idx, config, rev_vocab=None, key_vocab=None):
        super(NLLEntropy, self).__init__()
        self.padding_idx = padding_idx if padding_idx is not None else -100
        self.avg_type = config.avg_type

        if rev_vocab is None or key_vocab is None:
            self.weight = None
        else:
            self.logger.info("Use extra cost for key words")
            weight = np.ones(len(rev_vocab))
            for key in key_vocab:
                weight[rev_vocab[key]] = 10.0
            self.weight = cast_type(torch.from_numpy(weight), FLOAT,
                                    config.use_gpu)

    def forward(self, net_output, labels):
        batch_size = net_output.size(0)
        input = net_output.view(-1, net_output.size(-1))
        target = labels.view(-1)
        if self.avg_type is None:
            loss = F.nll_loss(input, target, size_average=False,
                              ignore_index=self.padding_idx,
                              weight=self.weight)
        elif self.avg_type == 'seq':
            loss = F.nll_loss(input, target, size_average=False,
                              ignore_index=self.padding_idx,
                              weight=self.weight)
            loss = loss / batch_size
        elif self.avg_type == 'real_word':
            loss = F.nll_loss(input, target, size_average=True,
                              ignore_index=self.padding_idx,
                              weight=self.weight, reduce=False)
            loss = loss.view(-1, net_output.size(1))
            loss = torch.sum(loss, dim=1)
            word_cnt = torch.sum(torch.sign(labels), dim=1).float()
            loss = loss/word_cnt
            loss = torch.mean(loss)
        elif self.avg_type == 'word':
            loss = F.nll_loss(input, target, size_average=True,
                              ignore_index=self.padding_idx,
                              weight=self.weight)
        else:
            raise ValueError("Unknown avg type")

        return loss


class NormKLLoss(_Loss):
    def __init__(self, unit_average=False):
        super(NormKLLoss, self).__init__()
        self.unit_average = unit_average

    def forward(self, recog_mu, recog_logvar, prior_mu, prior_logvar):
        # find the KL divergence between two Gaussian distribution
        loss = 1.0 + (recog_logvar - prior_logvar)
        loss -= torch.div(torch.pow(prior_mu - recog_mu, 2), torch.exp(prior_logvar))
        loss -= torch.div(torch.exp(recog_logvar), torch.exp(prior_logvar))
        if self.unit_average:
            kl_loss = -0.5 * torch.mean(loss, dim=1)
        else:
            kl_loss = -0.5 * torch.sum(loss, dim=1)
        avg_kl_loss = torch.mean(kl_loss)
        return avg_kl_loss


class GaussianKLLoss(_Loss):
    def __init__(self):
        super(GaussianKLLoss, self).__init__()

    def forward(self, mu, logvar, batch_size=None, unit_average=False):
        """
        0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        """
        y_kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)

        if unit_average:
            return torch.mean(y_kl)
        else:
            return torch.sum(y_kl)/batch_size


class PPLLoss(_Loss):
    logger = logging.getLogger()
    def __init__(self, config, vocab=None, key_vocab=None, ignore_vocab=None):
        super(PPLLoss, self).__init__()
        self.weight = None
        self.ignore = None
        if vocab is not None:
            if key_vocab is not None:
                self.logger.info("Use extra cost for key words")
                weight = np.ones(len(vocab))
                for key_w in key_vocab.values():
                    weight[vocab.token2id.get(key_w)] = 10.0
                self.weight = cast_type(torch.from_numpy(weight), FLOAT,
                                        config.use_gpu)
            if ignore_vocab is not None:
                self.logger.info("Use extra vocab for ignore words")
                ignore = np.ones(len(vocab))
                for ignore_w in ignore_vocab.values():
                    ignore[vocab.token2id.get(ignore_w)] = 0.0
                self.ignore = cast_type(torch.from_numpy(ignore), FLOAT,
                                        config.use_gpu)


    def forward(self, log_qy, target, batch_size=None, unit_average=False):
        """
        - py * log(q(y))
        """
        if self.weight is not None and self.ignore is None:
            y_ppl = - torch.sum(self.weight * target * log_qy, dim=1)
        elif self.ignore is not None and self.weight is None:
            y_ppl = - torch.sum(self.ignore * target * log_qy, dim=1)
        elif self.weight is not None and self.ignore is not None:
            y_ppl = - torch.sum(self.weight * self.ignore * target * log_qy, dim=1)
        else:
            y_ppl = - torch.sum(target * log_qy, dim=1)
        if unit_average:
            return torch.mean(y_ppl)
        else:
            return torch.sum(y_ppl) / batch_size


class CatKLLoss(_Loss):
    def __init__(self):
        super(CatKLLoss, self).__init__()

    def forward(self, log_qy, log_py, batch_size=None, unit_average=False):
        """
        qy * log(q(y)/p(y))
        """
        if log_qy.dim() > 2:
            log_qy = log_qy.squeeze()

        qy = torch.exp(log_qy)
        y_kl = torch.sum(qy * (log_qy - log_py), dim=-1)
        if unit_average:
            return torch.mean(y_kl)
        else:
            return torch.sum(y_kl)/batch_size


class CrossEntropyoss(_Loss):
    def __init__(self):
        super(CrossEntropyoss, self).__init__()

    def forward(self, log_qy, log_py, batch_size=None, unit_average=False):
        """
        -qy log(qy) + qy * log(q(y)/p(y))
        """
        if log_qy.dim() > 2:
            log_qy = log_qy.squeeze()
        qy = torch.exp(log_qy)
        h_q = torch.sum(-1 * log_qy * qy, dim=1)
        kl_qp = torch.sum(qy * (log_qy - log_py), dim=1)
        cross_ent = h_q + kl_qp
        if unit_average:
            return torch.mean(cross_ent)
        else:
            return torch.sum(cross_ent)/batch_size


class Entropy(_Loss):
    def __init__(self):
        super(Entropy, self).__init__()

    def forward(self, log_qy, batch_size=None, unit_average=False):
        """
        -qy log(qy)
        """
        if log_qy.dim() > 2:
            log_qy = log_qy.squeeze()
        qy = torch.exp(log_qy)
        h_q = torch.sum(-1 * log_qy * qy, dim=-1)
        if unit_average:
            return torch.mean(h_q)
        else:
            return torch.sum(h_q) / batch_size
