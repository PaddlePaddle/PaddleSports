# Modified from https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/dist_utils.py  # noqa: E501
import functools

def master_only(func):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # rank, _ = get_dist_info()
        rank = 0
        if rank == 0:
            return func(*args, **kwargs)

    return wrapper
