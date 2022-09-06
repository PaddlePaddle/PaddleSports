import paddle
import modules.losses as loss
from misc.rewards import get_self_critical_reward


class LossWrapper(paddle.nn.Layer):
    def __init__(self, model, opt):
        super(LossWrapper, self).__init__()
        self.opt = opt
        self.model = model
        self.crit = loss.LanguageModelCriterion()
        self.rl_crit = loss.RewardCriterion()

    def forward(self, fc_feats, att_feats, labels, masks, att_masks, gts, gt_indices,
                sc_flag):
        out = {}
        if not sc_flag:
            loss = self.crit(self.model(fc_feats, att_feats, labels, att_masks), labels[:, 1:], masks[:, 1:])
        else:
            self.model.eval()
            with paddle.no_grad():
                greedy_res, _ = self.model(fc_feats, att_feats, att_masks, mode='sample')
            self.model.train()
            gen_result, sample_logprobs = self.model(fc_feats, att_feats, att_masks, opt={'sample_method': 'sample',
                                                                                          'sample_n': self.opt.sample_n}, mode='sample')
            gts = [gts[_] for _ in gt_indices.tolist()]
            reward = get_self_critical_reward(greedy_res, gts, gen_result, self.opt)
            reward = paddle.to_tensor(reward)
            loss = self.rl_crit(sample_logprobs, gen_result, reward)
            out['reward'] = reward[:, 0].mean()
        out['loss'] = loss
        return out