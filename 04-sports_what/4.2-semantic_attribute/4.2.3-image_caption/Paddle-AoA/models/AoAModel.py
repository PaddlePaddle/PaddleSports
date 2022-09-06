# Implementation for paper 'Attention on Attention for Image Captioning'
# https://arxiv.org/abs/1908.06954

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from .AttModel import AttModel, pack_wrapper
from .TransformerModel import LayerNorm, attention, clones, SublayerConnection, PositionwiseFeedForward


class GLU(nn.Layer):
    """Applies the gated linear unit function."""
    def __init__(self, dim=-1):
        super(GLU, self).__init__()
        self.dim = dim

    def forward(self, input):
        return F.glu(input, axis=self.dim)


class MultiHeadedDotAttention(nn.Layer):
    def __init__(self, h, d_model, dropout=0.1, project_k_v=1, do_aoa=0, norm_q=0, dropout_aoa=0.3):
        super(MultiHeadedDotAttention, self).__init__()

        assert d_model % h == 0

        # We assume the dims of K and V are equal
        self.d_k = d_model // h
        self.h = h

        # Do we need to do linear projections on K and V?
        self.project_k_v = project_k_v

        # normalize the query?
        if norm_q:
            self.norm = LayerNorm(d_model)
        else:
            self.norm = lambda x: x

        self.linears = clones(module=nn.Linear(d_model, d_model), N=1 + 2 * project_k_v)

        # apply aoa after attention?
        self.use_aoa = do_aoa
        if self.use_aoa:
            self.aoa_layer = nn.Sequential(nn.Linear(2 * d_model, 2 * d_model), GLU())
            # dropout to the input of AoA layer
            if dropout_aoa > 0:
                self.dropout_aoa = nn.Dropout(p=dropout_aoa)
            else:
                self.dropout_aoa = lambda x: x

        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, value, key, mask=None):
        if mask is not None:
            if len(mask.shape) == 2:
                mask = paddle.unsqueeze(mask, axis=-2)
            # same mask applied to all h heads
            mask = paddle.unsqueeze(mask, axis=1)

        single_query = 0
        if len(query.shape) == 2:
            single_query = 1
            query = paddle.unsqueeze(query, axis=1)

        n_batch = query.shape[0]
        query = self.norm(query)

        # do all the linear projections in batch from d_model => h x d_k
        if self.project_k_v == 0:
            query_ = self.linears[0](query).reshape((n_batch, -1, self.h, self.d_k)).transpose((0, 2, 1, 3))
            key_ = key.reshape((n_batch, -1, self.h, self.d_k)).transpose((0, 2, 1, 3))
            value_ = value.reshape((n_batch, -1, self.h, self.d_k)).transpose((0, 2, 1, 3))
        else:
            query_, key_, value_ = \
                [l(x).reshape((n_batch, -1, self.h, self.d_k)).transpose((0, 2, 1, 3))
                 for l, x in zip(self.linears, (query, key, value))]

        x, self.attn = attention(query_, key_, value_, mask=mask, dropout=self.dropout)
        x = x.transpose([0, 2, 1, 3]).reshape((n_batch, -1, self.h * self.d_k))

        if self.use_aoa:
            # apply AoA
            x = self.aoa_layer(self.dropout_aoa(paddle.concat([x, query], axis=-1)))

        if single_query:
            x = paddle.squeeze(x, axis=1)
        return x


class AoA_Refiner_Layer(nn.Layer):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(AoA_Refiner_Layer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.use_ff = 0
        if self.feed_forward is not None:
            self.use_ff = 1
        self.sublayer = clones(SublayerConnection(size, dropout), 1 + self.use_ff)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[-1](x, self.feed_forward) if self.use_ff else x


class AoA_Refiner_Core(nn.Layer):
    def __init__(self, opt):
        super(AoA_Refiner_Core, self).__init__()
        attn = MultiHeadedDotAttention(opt.num_heads, opt.rnn_size, project_k_v=1, do_aoa=opt.refine_aoa,
                                       norm_q=0, dropout_aoa=getattr(opt, 'dropout_aoa', 0.3))
        layer = AoA_Refiner_Layer(opt.rnn_size, attn, PositionwiseFeedForward(opt.rnn_size, 2048, 0.1) if opt.use_ff else None, 0.1)
        self.layers = clones(layer, 4)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class AoA_Decoder_Core(nn.Layer):
    def __init__(self, opt):
        super(AoA_Decoder_Core, self).__init__()
        self.drop_prob_lm = opt.drop_prob_lm
        self.d_model = opt.rnn_size
        self.use_ctx_drop = getattr(opt, 'ctx_drop', 0)
        self.out_res = getattr(opt, 'out_res', 0)
        self.att_lstm = nn.LSTMCell(opt.input_encoding_size + opt.rnn_size, opt.rnn_size)
        self.out_drop = nn.Dropout(self.drop_prob_lm)

        # AoA layer
        self.att2ctx = nn.Sequential(
            nn.Linear(self.d_model + opt.rnn_size, 2 * opt.rnn_size), GLU())

        self.attention = MultiHeadedDotAttention(opt.num_heads, opt.rnn_size, project_k_v=0, do_aoa=0, norm_q=1)

        if self.use_ctx_drop:
            self.ctx_drop = nn.Dropout(self.drop_prob_lm)
        else:
            self.ctx_drop = lambda x: x

    def forward(self, xt, mean_feats, att_feats, p_att_feats, state, att_masks=None):
        att_lstm_input = paddle.concat([xt, mean_feats + self.ctx_drop(state[0][1])], axis=1)

        _, (h_att, c_att) = self.att_lstm(att_lstm_input, (state[0][0], state[1][0]))

        att = self.attention(h_att,
                             paddle.slice(p_att_feats, axes=[2], starts=[0], ends=[self.d_model]),
                             paddle.slice(p_att_feats, axes=[2], starts=[self.d_model], ends=[self.d_model * 2]),
                             att_masks)

        ctx_input = paddle.concat([att, h_att], axis=1)
        output = self.att2ctx(ctx_input)

        state = (paddle.stack([h_att, output]), paddle.stack([c_att, state[1][1]]))

        if self.out_res:
            # add residual connection
            output = output + h_att

        output = self.out_drop(output)
        return output, state


class AoAModel(AttModel):
    def __init__(self, opt):
        super(AoAModel, self).__init__(opt)
        self.num_layers = 2
        # mean pooling
        self.use_mean_feats = getattr(opt, 'mean_feats', 1)

        del self.ctx2att
        self.ctx2att = nn.Linear(opt.rnn_size, 2 * opt.rnn_size)

        if self.use_mean_feats:
            del self.fc_embed
        if opt.refine:
            self.refiner = AoA_Refiner_Core(opt)
        else:
            self.refiner = lambda x, y: x
        self.core = AoA_Decoder_Core(opt)

    def _prepare_feature(self, fc_feats, att_feats, att_masks):
        att_feats, att_masks = self.clip_att(att_feats, att_masks)

        # embed att feats
        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)
        att_feats = self.refiner(att_feats, att_masks)

        if self.use_mean_feats:
            # meaning pooling
            if att_masks is None:
                mean_feats = paddle.mean(att_feats, axis=1)
            else:
                mean_feats = (paddle.sum(att_feats * att_masks.unsqueeze(-1), 1) / paddle.sum(att_masks.unsqueeze(-1), 1))
        else:
            mean_feats = self.fc_embed(fc_feats)

        # Project the attention feats first to reduce memory and computation.
        p_att_feats = self.ctx2att(att_feats)

        return mean_feats, att_feats, p_att_feats, att_masks
