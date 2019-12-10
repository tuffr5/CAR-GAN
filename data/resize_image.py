from PIL import Image
import time
import os
import re
from torchvision import transforms
import random
import numpy as np

re_digits = re.compile(r'(\d+)')


def embedded_numbers(s):
    pieces = re_digits.split(s)
    pieces[1::2] = map(int, pieces[1::2])
    return pieces


def sort_string(lst):
    return sorted(lst, key=embedded_numbers)


def resize_crop(source, dest):
    image =  Image.open(source)
    params = get_params(image.size)
    transform = get_transform(params)
    image = transform(image)
    image.save(dest, image.format, quality=95)


def get_params(size):
    w, h = size
    new_h = h
    new_w = w
    new_h = new_w = 286

    x = random.randint(0, np.maximum(0, new_w - 256))
    y = random.randint(0, np.maximum(0, new_h - 256))

    return {'crop_pos': (x, y)}


def get_transform(params=None, method=Image.BICUBIC):
    transform_list = []
    osize = [286, 286]
    transform_list.append(transforms.Resize(osize, method))

    if params is None:
        transform_list.append(transforms.RandomCrop(256))
    else:
        transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], 256)))

    return transforms.Compose(transform_list)


def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img

def move_files(from_dir, to_dir):
    from_len = len(from_dir)
    assert os.path.isdir(from_dir), '%s is not a valid directory' % from_dir
    for root, _, fnames in sorted(os.walk(from_dir)):
        if fnames:
            for fname in sort_string(fnames):
                source_path = os.path.join(root, fname)
                des_path = os.path.join(to_dir, root[from_len:], fname)
                resize_crop(source_path, des_path)


def list_structure(from_dir, to_dir):
    if os.path.exists(to_dir):
        des_dir = []
        from_len = len(from_dir)
        assert os.path.isdir(from_dir), '%s is not a valid directory' % from_dir
        for root, dirs, _ in sorted(os.walk(from_dir)):
            for dir in dirs:
                path_truncate = root[from_len:]
                path = os.path.join(to_dir, path_truncate, dir)
                des_dir.append(path)
        return des_dir
    else:
        return None


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)



def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == '__main__':
    start_time = time.time()
    source = '~/Downloads/Sub-URMP/img/'
    destination = '~/Downloads/Sub-URMP/img_256/'
    mkdir(destination)
    destination_dir = list_structure(from_dir=source, to_dir=destination)
    mkdirs(paths=destination_dir)
    move_files(source, destination)
    print("The total time is %s" % (time.time() - start_time))
