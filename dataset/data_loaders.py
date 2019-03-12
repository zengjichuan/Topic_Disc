from __future__ import print_function
import numpy as np
import copy
from utils import Pack
from dataset.dataloader_bases import DataLoader

# Twitter Conversation
class TCDataLoader(DataLoader):
    def __init__(self, name, data, vocab_size, config):
        super(TCDataLoader, self).__init__(name, fix_batch=config.fix_batch)
        self.name = name
        self.vocab_size = vocab_size
        bow_data, m_cnt, w_cnt = data
        self.data = self.flatten_dialog(bow_data, config.window_size)
        self.data_size = len(self.data)
        if config.fix_batch:
            all_ctx_lens = [len(d.context) for d in self.data]
            self.indexes = list(np.argsort(all_ctx_lens))[::-1]
        else:
            self.indexes = list(range(len(self.data)))

    def flatten_dialog(self, data, window_size):
        results = []
        for dialog in data:
            for i in range(len(dialog)):
                c_id = i
                s_id = max(0, c_id - window_size//2)
                e_id = min(len(dialog), s_id + window_size)
                target = copy.copy(dialog[i])
                contexts = []
                for turn in dialog[s_id:e_id]:
                    contexts.append(turn)
                results.append(Pack(context=contexts, target=target))
        return results

    def _prepare_batch(self, selected_index):
        rows = [self.data[idx] for idx in selected_index]
        # input_context, context_lens, floors, topics, a_profiles, b_Profiles, outputs, output_lens
        context_lens, context_utts, target_utts, target_lens = [], [], [], []
        metas = []
        hashtags = []
        for row in rows:
            ctx = row.context
            target = row.target

            target_utt = target.utt
            context_lens.append(len(ctx))
            context_utts.append([turn.utt for turn in ctx])

            target_utts.append(target_utt)
            target_lens.append(len(target_utt))
            hashtags.append(target.hashtag)
            metas.append(target.meta)

        vec_context_lens = np.array(context_lens)
        vec_context = np.zeros((len(vec_context_lens), np.max(vec_context_lens),
                                self.vocab_size), dtype=np.int32)
        vec_targets = np.zeros((len(vec_context_lens), self.vocab_size), dtype=np.int32)
        vec_target_lens = np.array(target_lens)

        for b_id in range(len(vec_context_lens)):
            vec_targets[b_id, :] = self._bow2vec(target_utts[b_id], self.vocab_size)
            # fill the context tensor
            new_array = np.empty((vec_context_lens[b_id], self.vocab_size))
            new_array.fill(0)
            for i, row in enumerate(context_utts[b_id]):
                new_array[i, :] = self._bow2vec(row, self.vocab_size)
            vec_context[b_id, 0:vec_context_lens[b_id], :] = new_array

        return Pack(contexts=vec_context, context_lens=vec_context_lens,
                    targets=vec_targets, targets_lens=vec_target_lens,
                    metas=metas, hashtags=hashtags)


    def _bow2vec(self, bow, vec_size):
        vec = np.zeros(vec_size, dtype=np.int32)
        for id, val in bow:
            vec[id] = val
        return vec

