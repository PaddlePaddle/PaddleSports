from utils.logger import get_root_logger
from .rrdbnet_arch import RRDBNet
from .discriminator_arch import UNetDiscriminatorSN



def build_network(opt):
    net_type = opt["type"]
    if net_type=="RRDBNet":
        net = RRDBNet(num_in_ch=opt["num_in_ch"], num_out_ch=opt["num_out_ch"],
                      num_feat=opt["num_feat"], num_block=opt["num_block"],
                      num_grow_ch=opt["num_grow_ch"], scale=opt["scale"])
    elif net_type=="UNetDiscriminatorSN":
        net = UNetDiscriminatorSN(opt["num_in_ch"], num_feat=opt["num_feat"], skip_connection=opt["skip_connection"])
    else:
        raise KeyError(f"No archs named '{net_type}' found!")
    logger = get_root_logger()
    logger.info(f'Network [{net.__class__.__name__}] is created.')
    return net
