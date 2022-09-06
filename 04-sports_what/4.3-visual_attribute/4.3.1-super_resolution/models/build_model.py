from utils.logger import get_root_logger
from models.realesrgan_model import RealESRGANModel
from models.realesrnet_model import RealESRNetModel



def build_model(opt):
    """Build model from options.

    Args:
        opt (dict): Configuration. It must contain:
            model_type (str): Model type.
    """
    model_type = opt["model_type"]
    if model_type=='RealESRNetModel':
        model = RealESRNetModel(opt)
    elif model_type=='RealESRGANModel':
        model = RealESRGANModel(opt)
    else:
        raise KeyError(f"No model named '{model_type}' found!")

    logger = get_root_logger()
    logger.info(f'Model [{model.__class__.__name__}] is created.')
    return model



