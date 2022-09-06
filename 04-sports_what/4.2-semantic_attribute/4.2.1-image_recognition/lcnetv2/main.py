
# output raw json
# per frame ocr result

import argparse
from lcnet_main import LCNet_main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--class_num',type=int,default=10)
    parser.add_argument('--epoches',type=int,default=100)
    parser.add_argument('--use_pretrained', type=str,default= "")#PPLCNetV2_base_pretrained_without_fc.pdparams
    parser.add_argument('--data_dir', type=str,default="./data/d/Sports10/")
    parser.add_argument('--BATCH_SIZE', type=int,default=32) 
    parser.add_argument('--load_pretrain_model', type=str,default="")
    parser.add_argument('--output_model_dir', type=str,default="model")
    parser.add_argument('--is_train',type=int,default=1)
    parser.add_argument("--output_log_dir",type=str,default="./log")


    lcnet = LCNet_main(parser.parse_args())
    lcnet.main()
