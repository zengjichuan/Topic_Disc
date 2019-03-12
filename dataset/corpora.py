# -*- coding: utf-8 -*-
from __future__ import unicode_literals  # at top of module
from collections import Counter
import numpy as np
import json
from utils import get_tokenize, get_chat_tokenize, missingdict, Pack
import logging
import os
import re
from nltk.corpus import stopwords
import itertools
from collections import defaultdict
import copy
from gensim.parsing.preprocessing import STOPWORDS
from gensim.corpora import Dictionary

HT = "<hash>"
MEN = "<men>"
URL = "<url>"

TWITTER_STOPWORDS = ['amp', 'gt', 'lt', 'll']

class TwitterCorpus(object):
    logger = logging.getLogger(__name__)
    def __init__(self, config):
        self.config = config
        self._path = config.data_dir[0]
        self.tokenize = get_chat_tokenize()
        self.train_corpus, self.test_corpus, self.valid_corpus = self._read_file(os.path.join(self._path))
        self._build_vocab(config.max_vocab_cnt)
        print("Done loading corpus")

    def _process_dialog(self, data):
        new_dialog = []
        all_lens = []
        all_dialog_lens = []

        for raw_dialog in data:
            dialog = []
            for i, turn in enumerate(raw_dialog['text_lst']):
                utt = raw_dialog['text_lst'][i]
                utt = self.tokenize(utt.lower())
                all_lens.append(len(utt))
                dialog.append(Pack(utt=utt, hashtag=raw_dialog['hashtag_lst'][i], meta=raw_dialog['meta_lst'][i]))

            all_dialog_lens.append(len(dialog))
            new_dialog.append(dialog)

        print("Max utt len %d, mean utt len %.2f" % (
            np.max(all_lens), float(np.mean(all_lens))))
        print("Max dialog len %d, mean dialog len %.2f" % (
            np.max(all_dialog_lens), float(np.mean(all_dialog_lens))))
        return new_dialog

    def _build_vocab(self, max_vocab_cnt):
        all_words = []
        for dialog in self.train_corpus:
            for turn in dialog:
                all_words.append(turn.utt)

        self.vocab_bow = Dictionary(all_words)
        raw_vocab_size = len(self.vocab_bow)
        raw_wc = np.sum(list(self.vocab_bow.dfs.values()))

        # build useless stopwords vocab (e.g, very few words, single ascii words, some punctuation ,."')
        self.vocab_bow.filter_extremes(no_below=20)
        self.vocab_bow.filter_extremes(keep_n=max_vocab_cnt)
        bad_ids = [HT, MEN, URL] + TWITTER_STOPWORDS
        self.vocab_bow.filter_tokens(list(map(self.vocab_bow.token2id.get, bad_ids)))
        len_1_words = list(filter(lambda w: len(w) == 1 and re.match(r"[\x00-\x7f]", w) and w not in ["?", "!", "\"", "i"] and True or False, self.vocab_bow.values()))
        self.vocab_bow.filter_tokens(list(map(self.vocab_bow.token2id.get, len_1_words)))
        self.vocab_bow.compactify()
        # here we keep stopwords and some meaningful punctuations
        non_stopwords = filter(lambda w: re.match(r"^(?=.*[a-zA-Z\d])[\w\d_-]*$", w) and w not in STOPWORDS and True or False, self.vocab_bow.values())
        self.vocab_bow_stopwords = copy.deepcopy(self.vocab_bow)
        self.vocab_bow_stopwords.filter_tokens(map(self.vocab_bow_stopwords.token2id.get, non_stopwords))
        self.vocab_bow_stopwords.compactify()
        self.vocab_bow_non_stopwords = copy.deepcopy(self.vocab_bow)
        self.vocab_bow_non_stopwords.filter_tokens(map(self.vocab_bow_non_stopwords.token2id.get, self.vocab_bow_stopwords.values()))
        self.vocab_bow_non_stopwords.compactify()
        remain_wc = np.sum(list(self.vocab_bow.dfs.values()))
        min_count = np.min(list(self.vocab_bow.dfs.values()))
        # create vocabulary list sorted by count
        print("Load corpus with train size %d, valid size %d, "
              "test size %d raw vocab size %d vocab size %d at cut_off %d OOV rate %f"
              % (len(self.train_corpus), len(self.valid_corpus),
                 len(self.test_corpus),
                 raw_vocab_size, len(self.vocab_bow), min_count,
                 1 - float(remain_wc) / raw_wc))

    def _read_file(self, path):
        with open(path, 'rb') as f:
            data = json.load(f)
        return self._process_dialog(data["conv_train"]), self._process_dialog(data["conv_test"]), self._process_dialog(data["conv_valid"])

    def _sent2id(self, sent, vocab):
        return list(filter(lambda x: x is not None, [vocab.token2id.get(t) for t in sent]))

    def _sent2id_bow(self, sent, vocab):
        return vocab.doc2bow(sent)

    def _to_id_corpus(self, data, vocab):
        results = []
        word_cnt = 0
        msg_cnt = 0

        for dialog in data:
            temp = []
            # convert utterance and feature into numeric numbers
            for turn in dialog:
                id_turn = Pack(utt=self._sent2id(turn.utt, vocab), # need to plus 1 for padding
                               hashtag=turn.hashtag,
                               meta=turn.meta)
                if id_turn.utt:
                    temp.append(id_turn)
                    word_cnt += len(id_turn.utt)
                    msg_cnt += 1
            results.append(temp)
        print("Load seq with %d msgs, %d words" % (msg_cnt, word_cnt))
        return results, msg_cnt, word_cnt

    def _to_id_corpus_bow(self, data, vocab):
        results = []
        word_cnt = 0
        msg_cnt = 0

        for dialog in data:
            temp = []
            # convert utterance and feature into numeric numbers
            for turn in dialog:

                id_turn = Pack(utt=self._sent2id_bow(turn.utt, vocab),
                               hashtag=turn.hashtag,
                               meta=turn.meta)
                if id_turn.utt:     # filter empty utt
                    temp.append(id_turn)
                    word_cnt += np.sum([j for i, j in id_turn.utt])
                    msg_cnt += 1
            if temp:
                results.append(temp)
        print("Load bow with %d msgs, %d words" % (msg_cnt, word_cnt))
        return results, msg_cnt, word_cnt

    def get_corpus_bow(self, keep_stopwords=True):
        if keep_stopwords:
            vocab = self.vocab_bow
        else:
            vocab = self.vocab_bow_non_stopwords
        id_train = self._to_id_corpus_bow(self.train_corpus, vocab)
        id_valid = self._to_id_corpus_bow(self.valid_corpus, vocab)
        id_test = self._to_id_corpus_bow(self.test_corpus, vocab)
        return Pack(train=id_train, valid=id_valid, test=id_test, vocab_size=len(vocab))

    def get_corpus_seq(self, keep_stopwords=True):
        if keep_stopwords:
            vocab = self.vocab_bow
        else:
            vocab = self.vocab_bow_non_stopwords
        id_train = self._to_id_corpus(self.train_corpus, vocab)
        id_valid = self._to_id_corpus(self.valid_corpus, vocab)
        id_test = self._to_id_corpus(self.test_corpus, vocab)
        return Pack(train=id_train, valid=id_valid, test=id_test, vocab_size=len(vocab))

