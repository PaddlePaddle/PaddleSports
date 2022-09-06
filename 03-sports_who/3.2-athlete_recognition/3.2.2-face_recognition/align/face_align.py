from PIL import Image
from detector import detect_faces
from align_trans import get_reference_facial_points, warp_and_crop_face
import numpy as np
import os
from tqdm import tqdm
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "face alignment")
    parser.add_argument("-source_root", "--source_root", help = "specify your source dir", default = "./casia", type = str)
    parser.add_argument("-dest_root", "--dest_root", help = "specify your destination dir", default = "./aligned", type = str)
    parser.add_argument("-crop_size", "--crop_size", help = "specify size of aligned faces, align and crop with padding", default = 112, type = int)
    parser.add_argument("-hide_details","--hide_details",help ="output details",default=False,type=bool)
    args = parser.parse_args()

    source_root = args.source_root # specify your source dir
    dest_root = args.dest_root # specify your destination dir
    crop_size = args.crop_size # specify size of aligned faces, align and crop with padding
    scale = crop_size / 112.
    show_details = not args.hide_details
    reference = get_reference_facial_points(default_square = True) * scale

    cwd = os.getcwd() # delete '.DS_Store' existed in the source_root
    os.chdir(source_root)
    os.system("find . -name '*.DS_Store' -type f -delete")
    os.chdir(cwd)

    if not os.path.isdir(dest_root):
        os.mkdir(dest_root)
    for subfolder in tqdm(os.listdir(source_root)):
        dest_path = os.path.join(dest_root, subfolder)
        subfolder_path = os.path.join(source_root, subfolder)
        if not os.path.isdir(subfolder_path):
            if show_details:
                print('sikp {}'.format(subfolder_path))
            continue
        for image_name in os.listdir(subfolder_path):
            if show_details:
                print("Processing\t{}".format(os.path.join(source_root, subfolder, image_name)))
            img = Image.open(os.path.join(source_root, subfolder, image_name))
            try: # Handle exception
                _, landmarks = detect_faces(img)
            except Exception:
                if show_details:
                    print("{} is discarded due to exception!".format(os.path.join(source_root, subfolder, image_name)))
                continue
            if len(landmarks) == 0: # If the landmarks cannot be detected, the img will be discarded
                if show_details:
                    print("{} is discarded due to non-detected landmarks!".format(os.path.join(source_root, subfolder, image_name)))
                continue
            facial5points = [[landmarks[0][j], landmarks[0][j + 5]] for j in range(5)]
            warped_face = warp_and_crop_face(np.array(img), facial5points, reference, crop_size=(crop_size, crop_size))
            img_warped = Image.fromarray(warped_face)
            if image_name.split('.')[-1].lower() not in ['jpg', 'jpeg']: #not from jpg
                image_name = '.'.join(image_name.split('.')[:-1]) + '.jpg'
            if not os.path.isdir(dest_path):
                os.mkdir(dest_path)
            img_warped.save(os.path.join(dest_root, subfolder, image_name))
