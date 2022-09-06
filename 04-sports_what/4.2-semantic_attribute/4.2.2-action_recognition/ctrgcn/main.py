
# output raw json
# per frame ocr result

import argparse
from ctrgcn_main import CTRGCN_main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--window_size',type=int,default=1000)
    parser.add_argument('--epoches',type=int,default=100)
    parser.add_argument('--data_file', type=str,default="/home/aistudio/data/data104925/train_data.npy")
    parser.add_argument('--label_file', type=str,default="/home/aistudio/data/data104925/train_label.npy")
    parser.add_argument('--BATCH_SIZE', type=int,default=32) 
    parser.add_argument('--load_pretrain_model', type=str,default="")
    parser.add_argument('--output_model_dir', type=str,default="model")
    parser.add_argument('--is_train',type=int,default=1)
    parser.add_argument("--output_log_dir",type=str,default="./log")
    parser.add_argument("--data_mode",type = str,default = "joint",help="joint or bone")


    ctrgcn = CTRGCN_main(parser.parse_args())
    ctrgcn.main()
