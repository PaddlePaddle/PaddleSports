import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class RewardCriterion(nn.Layer):
    def __init__(self):
        super(RewardCriterion, self).__init__()

    def forward(self, input, seq, reward):
        input = input.reshape([-1])
        reward = reward.reshape([-1])
        mask = (seq > 0).astype('float32')
        fill_ones = paddle.ones([mask.shape[0], 1], dtype='float32')
        mask = paddle.concat([fill_ones, mask[:, :-1]], 1).reshape([-1])
        output = - input * reward * mask
        output = paddle.sum(output) / paddle.sum(mask)

        return output


class LanguageModelCriterion(nn.Layer):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):
        # truncate to the same size
        target = target[:, :input.shape[1]]
        mask = mask[:, :input.shape[1]]
        target = F.one_hot(target, input.shape[-1])
        output = -input.multiply(target).sum(-1) * mask
        output = paddle.sum(output) / paddle.sum(mask)
        return output


class LabelSmoothing(nn.Layer):
    "Implement label smoothing."

    def __init__(self, vocab_size, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='none')
        self.vocab_size = vocab_size + 1
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, input, target, mask):
        # truncate to the same size
        target = target[:, :input.shape[1]]
        mask = mask[:, :input.shape[1]]
        input = input.reshape((-1, input.shape[-1]))
        target = target.reshape((-1,))
        mask = mask.reshape((-1,))

        size = input.shape[1]
        target_one_hot = F.one_hot(target, num_classes=self.vocab_size)
        x = paddle.full(target_one_hot.shape, dtype=target_one_hot.dtype, fill_value=self.confidence)
        y = paddle.full(target_one_hot.shape, dtype=target_one_hot.dtype, fill_value=self.smoothing / (size - 1))
        true_dist = paddle.where(target_one_hot != 0, x, y)

        return (self.criterion(input, true_dist).sum(1) * mask).sum() / mask.sum()
