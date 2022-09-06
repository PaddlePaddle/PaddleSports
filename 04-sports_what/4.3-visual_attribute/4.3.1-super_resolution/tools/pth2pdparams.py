import sys
sys.path.append('.')
import os
import time

import paddle
import torch
from models.rrdbnet_arch import RRDBNet
from utils.logger import get_root_logger

def transfer(input_path, out_path):
    torch_dict = torch.load(input_path, map_location=torch.device('cpu'))
    print(torch_dict['params_ema'].keys())
    net_g = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    save_network(net_g, out_path, 'params', torch_dict)
    
    

def save_network(net, save_path, param_key='params', torch_dict=None):
        """Save networks.

        Args:
            net (nn.Module | list[nn.Module]): Network(s) to be saved.
            net_label (str): Network label.
            current_iter (int): Current iter number.
            param_key (str | list[str]): The parameter key(s) to save network.
                Default: 'params'.
        """

        net = net if isinstance(net, list) else [net]
        param_key = param_key if isinstance(param_key, list) else [param_key]
        assert len(net) == len(param_key), 'The lengths of net and param_key should be the same.'

        save_dict = {}
        for net_, param_key_ in zip(net, param_key):
            # net_ = self.get_bare_model(net_)
            state_dict = {}
            torch_dict = torch_dict['params_ema']
            for key, param in torch_dict.items():
                if key.startswith('module.'):  # remove unnecessary 'module.'
                    key = key[7:]
                state_dict[key] = param.cpu().detach().numpy()
            save_dict[param_key_] = state_dict

        # avoid occasional writing errors
        retry = 3
        while retry > 0:
            try:
                paddle.save(save_dict, save_path, protocol=4)
            except Exception as e:
                logger = get_root_logger()
                logger.warning(f'Save model error: {e}, remaining retry times: {retry - 1}')
                time.sleep(1)
            else:
                break
            finally:
                retry -= 1
        if retry == 0:
            logger.warning(f'Still cannot save {save_path}. Just ignore it.')

if __name__ == "__main__":
    input_path = r"weights\RealESRGAN_x4plus.pth"
    out_path = r"weights\RealESRGAN_x4plus.pdparams"
    transfer(input_path, out_path)
