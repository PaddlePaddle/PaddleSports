import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import copy
import math


def clones(module, N):
    "Produce N identical layers."
    return nn.LayerList([copy.deepcopy(module) for _ in range(N)])


class SublayerConnection(nn.Layer):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class LayerNorm(nn.Layer):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = paddle.create_parameter(shape=[features, ], dtype='float32',
                                           default_initializer=nn.initializer.Constant(value=1.))
        self.b_2 = paddle.create_parameter(shape=[features, ], dtype='float32',
                                           default_initializer=nn.initializer.Constant(value=0.))
        self.eps = eps

    def forward(self, x):
        mean = paddle.mean(x, axis=-1, keepdim=True)
        std = paddle.std(x, axis=-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class PositionwiseFeedForward(nn.Layer):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention."
    d_k = query.shape[-1]
    scores = paddle.matmul(query, key.transpose([0, 1, 3, 2])) / math.sqrt(d_k)

    if mask is not None:
        scores_mask = paddle.fluid.layers.fill_constant(shape=scores.shape, dtype=scores.dtype, value=-1e9)
        scores = paddle.where(paddle.broadcast_to(mask, shape=scores.shape) != 0, scores, scores_mask)
    p_attn = F.softmax(scores, axis=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    return paddle.matmul(p_attn, value), p_attn


