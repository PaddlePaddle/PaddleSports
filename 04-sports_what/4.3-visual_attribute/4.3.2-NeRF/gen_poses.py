from poses.pose_utils import gen_poses
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--factors', type=int, 
					default=None, help='input images factor')
parser.add_argument('--scenedir', type=str, default=r"satchel",
                    help='input scene directory')
args = parser.parse_args()


if __name__=='__main__':
    gen_poses(args.scenedir, args.factors)