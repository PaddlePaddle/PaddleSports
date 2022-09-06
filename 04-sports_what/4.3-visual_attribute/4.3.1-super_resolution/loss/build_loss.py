from .losses import L1Loss, GANLoss
from utils.logger import get_root_logger
from ppgan.models.criterions import PerceptualLoss

def build_loss(opt):
    loss_type = opt["type"]
    if loss_type == 'L1Loss':
        loss = L1Loss(opt)
    elif loss_type == 'PerceptualLoss':
        loss = PerceptualLoss(layer_weights=opt['layer_weights'],use_input_norm=opt['use_input_norm'],
                              perceptual_weight=opt['perceptual_weight'],style_weight=opt['style_weight'],
                              vgg_type=opt['vgg_type'],norm_img=opt['norm_img'],criterion=opt['criterion'])
    elif loss_type == 'GANLoss':
        loss = GANLoss(opt["gan_type"], real_label_val=opt["real_label_val"],
                       fake_label_val=opt["fake_label_val"], loss_weight=opt["loss_weight"])
        # loss = GANLoss(opt)
    else:
        raise KeyError(f"No model named '{loss_type}' found!")

    logger = get_root_logger()
    logger.info(f'Loss [{loss.__class__.__name__}] is created.')
    return loss
