import os
import zipfile

import argparse

def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            # print(1, file)
            if file.endswith('gt.txt') or file.endswith('seqinfo.ini'):
                # print(file)
                ziph.write(os.path.join(root, file), 
                        os.path.relpath(os.path.join(root, file), 
                                        os.path.join(path, '..')))

def main(args):
    with zipfile.ZipFile('gt.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipdir(args.folder, zipf)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--folder', type=str, default = './SNMOT/images/test')

    args = parser.parse_args()
    main(args)