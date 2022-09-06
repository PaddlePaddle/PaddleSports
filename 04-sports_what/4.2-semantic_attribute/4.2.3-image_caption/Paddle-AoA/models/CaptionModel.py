from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class CaptionModel(nn.Layer):
    def __init__(self):
        super(CaptionModel, self).__init__()

    # implements beam search
    # calls beam_step and returns the final set of beams
    # augments log-probabilities with diversity terms when number of groups > 1

    def forward(self, *args, **kwargs):
        mode = kwargs.get('mode', 'forward')
        if 'mode' in kwargs:
            del kwargs['mode']
        return getattr(self, '_' + mode)(*args, **kwargs)

    def beam_search(self, state, *args, opt=None):
        # args: tmp_fc_feats, tmp_att_feats, tmp_att_cnn_feats, tmp_p_att_feats, tmp_att_masks
        args = list(args)

        k = opt['beam_size']
        vocab_size = self.vocab_size + 1
        it = paddle.zeros([k], dtype='int64')  # k

        # Tensor to store top k sequences; now they're just <start>
        seqs = it.unsqueeze(1)  # (k, 1)

        # Tensor to store top k sequences' scores; now they're just 0
        top_k_scores = paddle.zeros([k, 1], dtype='int64')  # (k, 1)

        # Lists to store completed sequences, their alphas and scores
        complete_seqs = list()
        complete_seqs_scores = list()

        # Start decoding
        step = 1

        # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
        while True:
            scores, state = self.get_logprobs_state(it, *args, state)

            # Add
            scores = top_k_scores.expand_as(scores) * step + scores  # (s, vocab_size)
            avg_scores = scores / step  # (s, vocab_size)

            # For the first step, all k points will have the same scores (since same k previous words, h, c)
            if step == 1:
                top_k_scores, top_k_words = avg_scores[0].topk(k, 0, True, True)  # (s)
            else:
                # Unroll and find top scores, and their unrolled indices
                top_k_scores, top_k_words = avg_scores.reshape([-1]).topk(k, 0, True, True)  # (s)

            # Convert unrolled indices to actual indices of scores
            prev_word_inds = top_k_words // vocab_size  # (s)
            next_word_inds = top_k_words % vocab_size  # (s)

            # Add new words to sequences
            seqs = paddle.concat([seqs[prev_word_inds.unsqueeze(1)], next_word_inds.unsqueeze(1)], 1)  # (s, step+1)

            # Which sequences are incomplete (didn't reach <end>)?
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                               next_word != 0]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            incomplete_inds = paddle.to_tensor(incomplete_inds).unsqueeze(1)
            complete_inds = paddle.to_tensor(complete_inds).unsqueeze(1)

            # Set aside complete sequences
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds])
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)  # reduce beam length accordingly

            # Proceed with incomplete sequences
            if k == 0:
                break

            seqs = seqs[incomplete_inds]

            nstate = []
            for s in range(len(state)):
                pstate = []
                for j, l in enumerate(prev_word_inds[incomplete_inds]):
                    ns = state[s][:, l]
                    pstate.append(ns)
                pstate = paddle.stack(pstate, axis=1)
                nstate.append(pstate)
            state = nstate

            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            it = next_word_inds[incomplete_inds]

            args = [args[i][incomplete_inds] for i in range(len(args))]

            # Break if things have been going on too long
            if step >= self.seq_length:
                complete_seqs.extend(seqs[incomplete_inds])
                complete_seqs_scores.extend(top_k_scores[incomplete_inds])
                break
            step += 1

        i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[i].numpy()
        tokens = list(filter(lambda x: x != 0, seq))
        tokens = paddle.to_tensor(tokens)
        out = [{'seq': tokens}]

        return out

    def sample_next_word(self, logprobs, sample_method, temperature):
        if sample_method == 'greedy':
            sampleLogprobs, it = paddle.topk(logprobs, k=1, axis=1)
            sampleLogprobs = sampleLogprobs.squeeze(-1)
            it = it.squeeze(-1)
        else:
            if temperature == 1.0:
                prob_prev = paddle.exp(logprobs)  # fetch prev distribution: shape Nx(M+1)
            else:
                # scale logprobs by temperature
                prob_prev = paddle.exp(logprobs / temperature)

            it = paddle.multinomial(prob_prev, 1).squeeze(-1)
            # gather the logprobs at sampled positions
            arget = F.one_hot(it, prob_prev.shape[-1])
            sampleLogprobs = logprobs.multiply(arget).sum(-1)

        return it, sampleLogprobs