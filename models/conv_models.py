# -*- coding: utf-8 -*-
import itertools

import numpy as np
import torch
import logging
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from models.encoders import MultiFC
import criterions
from utils import FLOAT, LONG, cast_type
from models.model_bases import BaseModel
from utils import Pack
import numbers

INFER = "infer"
TRAIN = "train"

class TDM(BaseModel):
    logger = logging.getLogger(__name__)
    def __init__(self, corpus, config):
        super(TDM, self).__init__(config)
        self.vocab_bow = corpus.vocab_bow
        self.vocab_bow_stopwords = corpus.vocab_bow_stopwords
        self.vocab_size = len(self.vocab_bow)
        if not hasattr(config, "freeze_step"):
            config.freeze_step = 6000

        # build mode here
        # x is for discourse
        self.x_encoder = MultiFC(self.vocab_size, config.hidden_size, config.d,
                                 num_hidden_layers=1, short_cut=True)

        self.x_generator = MultiFC(config.d, config.d, config.d,
                                   num_hidden_layers=0, short_cut=False)
        self.x_decoder = nn.Linear(config.d, self.vocab_size, bias=False)

        # context encoder
        # ctx is for topic
        self.ctx_encoder = MultiFC(self.vocab_size, config.hidden_size, config.hidden_size,
                                   num_hidden_layers=1, short_cut=True)
        self.q_z_mu, self.q_z_logvar = nn.Linear(config.hidden_size, config.k), nn.Linear(config.hidden_size, config.k)
        # cnn
        # self.ctx_encoder = CtxEncoder(config, utt_encoder=self.utt_encoder)
        self.ctx_generator = MultiFC(config.k, config.k, config.k, num_hidden_layers=0, short_cut=False)

        # decoder
        self.ctx_dec_connector = nn.Linear(config.k, config.k, bias=True)
        self.x_dec_connector = nn.Linear(config.d, config.d, bias=True)
        self.ctx_decoder = nn.Linear(config.k, self.vocab_size)

        self.decoder = nn.Linear(config.k + config.d, self.vocab_size, bias=False)

        # connector
        self.cat_connector = GumbelConnector()
        self.nll_loss = criterions.PPLLoss(self.config)
        self.nll_loss_filtered = criterions.PPLLoss(self.config, vocab=self.vocab_bow,
                                                    ignore_vocab=self.vocab_bow_stopwords)
        self.kl_loss = criterions.GaussianKLLoss()
        self.cat_kl_loss = criterions.CatKLLoss()
        self.entropy_loss = criterions.Entropy()
        self.reg_l1_loss = criterions.L1RegLoss(0.70)
        self.log_uniform_d = Variable(torch.log(torch.ones(1) / config.d))
        if self.use_gpu:
            self.log_uniform_d = self.log_uniform_d.cuda()

    def qdx_forward(self, tar_utts):
        qd_logits = self.x_encoder(tar_utts).view(-1, self.config.d)
        qd_logits_multi = qd_logits.repeat(self.config.d_size, 1, 1)
        sample_d_multi, d_ids_multi = self.cat_connector(qd_logits_multi, 1.0,
                                                         self.use_gpu, return_max_id=True)
        sample_d = sample_d_multi.mean(0)
        d_ids = d_ids_multi.view(self.config.d_size, -1).transpose(0, 1)

        return Pack(qd_logits=qd_logits, sample_d=sample_d, d_ids=d_ids)

    def pxy_forward(self, results):
        gen_d = self.x_generator(results.sample_d)
        x_out = self.x_decoder(gen_d)

        results['gen_d'] = gen_d
        results['x_out'] = x_out

        return results

    def qzc_forward(self, ctx_utts):
        ctx_out = F.tanh(self.ctx_encoder(ctx_utts))
        z_mu = self.q_z_mu(ctx_out)
        z_logvar = self.q_z_logvar(ctx_out)

        sample_z = self.reparameterize(z_mu, z_logvar)
        return Pack(sample_z=sample_z, z_mu=z_mu, z_logvar=z_logvar)

    def pcz_forward(self, results):
        gen_c = self.ctx_generator(results.sample_z)
        c_out = self.ctx_decoder(gen_c)

        results['gen_c'] = gen_c
        results['c_out'] = c_out

        return results

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def valid_loss(self, loss, batch_cnt=None):
        vae_x_loss = loss.vae_x_nll + loss.vae_x_kl
        vae_c_loss = loss.vae_c_nll + loss.vae_c_kl
        div_kl = loss.div_kl
        dec_loss = loss.nll

        if self.config.use_l1_reg:
            vae_c_loss += loss.l1_reg

        if batch_cnt is not None and batch_cnt > self.config.freeze_step:
            total_loss = dec_loss
            self.flush_valid = True
            for param in self.x_encoder.parameters():
                param.requires_grad = False
            for param in self.x_generator.parameters():
                param.requires_grad = False
            for param in self.x_decoder.parameters():
                param.requires_grad = False
            for param in self.ctx_encoder.parameters():
                param.requires_grad = False
            for param in self.ctx_generator.parameters():
                param.requires_grad = False
            for param in self.ctx_decoder.parameters():
                param.requires_grad = False
        else:
            total_loss = vae_x_loss + vae_c_loss + 0.001*div_kl

        return total_loss

    def forward(self, data_feed, mode=TRAIN, return_latent=False):
        batch_size = len(data_feed['targets'])

        ctx_utts = self.np2var(data_feed['contexts'], FLOAT)
        tar_utts = self.np2var(data_feed['targets'], FLOAT)

        # vae here
        vae_x_resp = self.pxy_forward(self.qdx_forward(tar_utts))

        # context encoder
        ctx_utts = ctx_utts.sum(1)  # merge contexts into one bow
        vae_c_resp = self.pcz_forward(self.qzc_forward(ctx_utts))

        # prior network (we can restrict the prior to stopwords and emotional words)

        # combine context topic and x discourse
        sample_d, d_ids = vae_x_resp.sample_d.detach(), vae_x_resp.d_ids.detach()
        sample_z = vae_c_resp.sample_z.detach()
        gen = torch.cat([self.x_dec_connector(sample_d), self.ctx_dec_connector(sample_z)], dim=1)
        dec_out = self.decoder(gen)

        # compute loss or return results
        if mode == INFER:
            return Pack(gen=gen, d_ids=d_ids)
        # vae-related losses
        log_qx = F.log_softmax(vae_x_resp.x_out, dim=1)
        log_qd = F.log_softmax(vae_x_resp.qd_logits, dim=1)
        vae_x_nll = self.nll_loss(log_qx, tar_utts, batch_size, unit_average=True)
        avg_log_qd = torch.exp(log_qd)
        avg_log_qd = torch.log(torch.mean(avg_log_qd, dim=0) + 1e-15)
        vae_x_kl = self.cat_kl_loss(avg_log_qd, self.log_uniform_d, batch_size, unit_average=True)

        log_qc = F.log_softmax(vae_c_resp.c_out, dim=1)
        vae_c_nll = self.nll_loss_filtered(log_qc, ctx_utts, batch_size, unit_average=True)
        vae_c_kl = self.kl_loss(vae_c_resp.z_mu, vae_c_resp.z_logvar, batch_size, unit_average=True)

        div_kl = - self.cat_kl_loss(log_qx, log_qc, batch_size, unit_average=True)  # maximize the kl loss

        # decoder loss
        log_dec = F.log_softmax(dec_out, dim=1)
        dec_nll = self.nll_loss(log_dec, tar_utts, batch_size, unit_average=True)

        # regularization loss
        if self.config.use_l1_reg:
            l1_reg = self.reg_l1_loss(self.ctx_decoder.weight, torch.zeros_like(self.ctx_decoder.weight))
        else:
            l1_reg = None

        results = Pack(nll=dec_nll, vae_x_nll=vae_x_nll, vae_x_kl=vae_x_kl, vae_c_nll=vae_c_nll,
                       vae_c_kl=vae_c_kl, l1_reg=l1_reg, div_kl=div_kl)
        if return_latent:
            results['gen'] = gen
            results['d_ids'] = d_ids
        return results

    def model_sel_loss(self, loss, batch_cnt):
        return self.valid_loss(loss, batch_cnt)


class GumbelConnector(nn.Module):
    def __init__(self):
        super(GumbelConnector, self).__init__()

    def sample_gumbel(self, logits, use_gpu, eps=1e-20):
        u = torch.rand(logits.size())
        sample = Variable(-torch.log(-torch.log(u + eps) + eps))
        sample = cast_type(sample, FLOAT, use_gpu)
        return sample

    def gumbel_softmax_sample(self, logits, temperature, use_gpu):
        """ Draw a sample from the Gumbel-Softmax distribution"""
        eps = self.sample_gumbel(logits, use_gpu)
        y = logits + eps
        return F.softmax(y / temperature, dim=y.dim()-1)

    def forward(self, logits, temperature, use_gpu, hard=False,
                return_max_id=False):
        """
        :param logits: [batch_size, n_class] unnormalized log-prob
        :param temperature: non-negative scalar
        :param hard: if True take argmax
        :return: [batch_size, n_class] sample from gumbel softmax
        """
        y = self.gumbel_softmax_sample(logits, temperature, use_gpu)
        _, y_hard = torch.max(y, dim=-1, keepdim=True)
        if hard:
            y_onehot = cast_type(Variable(torch.zeros(y.size())), FLOAT, use_gpu)
            y_onehot.scatter_(-1, y_hard, 1.0)
            y = y_onehot
        if return_max_id:
            return y, y_hard
        else:
            return y