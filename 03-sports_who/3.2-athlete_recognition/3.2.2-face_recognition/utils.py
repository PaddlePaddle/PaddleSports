import matplotlib.pyplot as plt
from datetime import datetime
import io,os
from PIL import Image
import numpy as np
import paddle
from paddle.vision import transforms
from tools.verification import evaluate

plt.switch_backend('agg')


def get_time():
    return (str(datetime.now())[:-10]).replace(' ', '-').replace(':', '-')


def separate_irse_bn_paras(modules):
    if not isinstance(modules, list):
        modules = [*modules.sublayers()]
    paras_only_bn = []
    paras_wo_bn = []
    for layer in modules:
        layer_info = str(layer.__class__)
        if 'backbones' in layer_info:
            continue
        if 'container' in layer_info:
            continue
        else:
            if 'BatchNorm' in str(layer.__class__):
                paras_only_bn.extend([*layer.parameters()])
            else:
                paras_wo_bn.extend([*layer.parameters()])

    return paras_only_bn, paras_wo_bn


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.shape[0]
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.equal(target.reshape((1, -1)).expand_as(pred))

    res = []
    for k in topk:
        correct_k = (correct.numpy())[:k].reshape((-1)).astype('float32').sum(0)
        res.append(correct_k * (100.0 / batch_size))

    return res


def warm_up_lr(batch, num_batch_warm_up, init_lr, optimizer):
    # print('\n warm_up_lr {}--->{}'.format(optimizer._learning_rate,batch * init_lr / num_batch_warm_up))
    optimizer._learning_rate = batch * init_lr / num_batch_warm_up
    return optimizer

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def schedule_lr(optimizer):
    # after warm_up_lr LinearWarmup replaced by learning_rate
    # print('\n schedule_lr {}--->{}'.format(optimizer._learning_rate,optimizer._learning_rate / 10.))
    optimizer._learning_rate = optimizer._learning_rate / 10.


def separate_resnet_bn_paras(modules):
    all_parameters = modules.parameters()
    paras_only_bn = []

    for pname, p in modules.named_parameters():
        if pname.find('bn') >= 0:
            paras_only_bn.append(p)

    paras_only_bn_id = list(map(id, paras_only_bn))
    paras_wo_bn = list(filter(lambda p: id(p) not in paras_only_bn_id, all_parameters))

    return paras_only_bn, paras_wo_bn


def hflip_batch(imgs_tensor):
    hfliped_imgs = paddle.empty_like(imgs_tensor)
    for i, img_ten in enumerate(imgs_tensor):
        hfliped_imgs[i] = transforms.hflip(img_ten)

    return hfliped_imgs


def l2_norm(input, axis=1):
    norm = paddle.norm(input, 2, axis, True)
    output = paddle.divide(input, norm)

    return output


def perform_val(epoch_id,embedding_size, batch_size, backbone, carray, issame, save_val_roc=False,nrof_folds=10, tta=True):
    backbone.eval()  # switch to evaluation mode
    idx = 0
    embeddings = np.zeros([len(carray), embedding_size])
    while idx + batch_size <= len(carray):
        # 1. read from raw data
        # batch = paddle.to_tensor(carray[idx:idx + batch_size][:, [2, 1, 0], :, :])
        # batch = paddle.to_tensor(carray[idx:idx + batch_size])

        # 2. get data from val dataloader
        batch = carray[idx:idx + batch_size]

        if tta:
            # ccropped = ccrop_batch(batch)
            # fliped = hflip_batch(ccropped)

            fliped = hflip_batch(batch)
            emb_batch = backbone(batch) + backbone(fliped)
            embeddings[idx:idx + batch_size] = l2_norm(emb_batch)
        else:
            embeddings[idx:idx + batch_size] = l2_norm(backbone(batch))
        idx += batch_size
    if idx < len(carray):
        batch = paddle.to_tensor(carray[idx:])
        if tta:
            # ccropped = ccrop_batch(batch)
            # fliped = hflip_batch(ccropped)
            fliped = hflip_batch(batch)
            emb_batch = backbone(batch) + backbone(fliped)
            embeddings[idx:idx + batch_size] = l2_norm(emb_batch)
        else:
            embeddings[idx:] = l2_norm(backbone(batch))

    tpr, fpr, accuracy, best_thresholds = evaluate(embeddings, issame, nrof_folds)
    if save_val_roc:
        buf = gen_plot(fpr, tpr)
        roc_curve = Image.open(buf)
        roc_curve.save("epoch_{}_val_roc.jpg".format(epoch_id))
    # roc_curve_tensor = transforms.ToTensor()(roc_curve)
    return accuracy.mean(), best_thresholds.mean()


def gen_plot(fpr, tpr):
    """Create a pyplot plot and save to buffer."""
    plt.figure()
    plt.xlabel("FPR", fontsize=14)
    plt.ylabel("TPR", fontsize=14)
    plt.title("ROC Curve", fontsize=14)
    plot = plt.plot(fpr, tpr, linewidth=2)
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    plt.close()

    return buf


def load_weight(model, weight_path, optimizer=None,downloaded = False):
    print('*'*60)
    print('Loading weight from pretrained ...')
    pdparam_path = weight_path
    if not os.path.exists(pdparam_path):
        raise ValueError("Model pretrain path {} does not "
                         "exists.".format(pdparam_path))

    param_state_dict = paddle.load(pdparam_path)
    model_dict = model.state_dict()
    model_weight = {}
    incorrect_keys = 0
    extended_layer = ['output_fc.weight','output_fc.bias','bn_output.weight','bn_output.bias','bn_output._mean','bn_output._variance']

    if downloaded:
        print('loading downloaded weight')
        for key in model_dict.keys():
            transform_key = 'backbone.'+key
            if transform_key in param_state_dict.keys():
                # set weight
                model_weight[key] = param_state_dict[transform_key]
            elif key in extended_layer:
                print('extenction layer:' + key)
                continue
            else:
                print('Unmatched key: {} with {}'.format(key,transform_key))
                incorrect_keys += 1
    else:
        print('loading checkpoint weight')
        for key in model_dict.keys():
            if key in param_state_dict.keys():
                # set weight
                model_weight[key] = param_state_dict[key]
            else:
                print('Unmatched key: {} of model in loaded checkpoint'.format(key))
                incorrect_keys += 1
    assert incorrect_keys == 0, "Load weight {} incorrectly, \
            {} keys unmatched, please check again.\n checkpoint param_state_dict keys: \n {}".format(weight_path,
                                                        incorrect_keys,param_state_dict.keys())
    print('Finished resuming model weights: {}'.format(pdparam_path))

    model.set_dict(model_weight)

    last_epoch = 0
    pdopt_name, ext = os.path.splitext(weight_path)
    if optimizer is not None and os.path.exists(pdopt_name + '.pdopt'):
        optim_state_dict = paddle.load(pdopt_name + '.pdopt')
        # to solve resume bug, will it be fixed in paddle 2.0
        for key in optimizer.state_dict().keys():
            if not key in optim_state_dict.keys():
                print('load optimizer key {}'.format(key))
                optim_state_dict[key] = optimizer.state_dict()[key]
        if 'last_epoch' in optim_state_dict:
            last_epoch = optim_state_dict.pop('last_epoch')
            print('load optimizer last_epoch {}'.format(last_epoch))
        optimizer.set_state_dict(optim_state_dict)
        print('load optimizer finished')
    print('*'*60)
    return last_epoch
