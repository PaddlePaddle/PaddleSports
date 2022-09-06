import os
import glob
import os.path as osp


def gen_image_list(dataPath, datType, image_list_root='./image_lists'):
    if not os.path.exists(image_list_root):
        os.mkdir(image_list_root)
    inputPath = f'{dataPath}/images/{datType}'
    pathList = glob.glob(inputPath + '/*')
    pathList = sorted(pathList)
    allImageList = []
    for pathSingle in pathList:
        imgList = sorted(glob.glob(osp.join(pathSingle, 'img1', '*.jpg')))
        for imgPath in imgList:
            allImageList.append(imgPath)
    image_list_fname = osp.join(image_list_root, f'{dataPath}.{datType}')
    with open(image_list_fname, 'w') as image_list_file:
        allImageListStr = str.join('\n', allImageList)
        image_list_file.write(allImageListStr)


if __name__ == '__main__':
    dataPath = 'SNMOT'
    datType = 'train'
    gen_image_list(dataPath, datType)
