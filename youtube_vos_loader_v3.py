# system modules
import os, argparse
from datetime import datetime



# basic pytorch modules
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision
from torchvision.datasets.vision import VisionDataset
import torchvision.transforms as transforms
from torchvision.datasets import UCF101
from Dataloader import Kittiloader
from dataset import DataGenerator

# custom utilities
from dataloader import MNIST_Dataset, KTH_Dataset
from convlstmnet import ConvLSTMNet
from contextvpnet import ContextVPNet
import matplotlib.image as imgblin
import skimage

from torch.nn.init import kaiming_normal_, constant_

import numpy as np
from torch.utils.data import Dataset, DataLoader

from PIL import Image

from typing import Any, Callable, cast, Dict, List, Optional, Tuple
import random

def has_file_allowed_extension(filename: str, extensions: Tuple[str, ...]) -> bool:
    """Checks if a file is an allowed extension.
    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)
    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)


def is_image_file(filename: str) -> bool:
    """Checks if a file is an allowed image extension.
    Args:
        filename (string): path to a file
    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def make_dataset(
    directory: str,
    num_of_frames: int,
    class_to_idx: Dict[str, int],
    extensions: Optional[Tuple[str, ...]] = None,
    is_valid_file: Optional[Callable[[str], bool]] = None,
) -> List[Tuple[str, int]]:
    instances = []
    directory = os.path.expanduser(directory)
    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x: str) -> bool:
            return has_file_allowed_extension(x, cast(Tuple[str, ...], extensions))
    is_valid_file = cast(Callable[[str], bool], is_valid_file)
    for target_class in sorted(class_to_idx.keys()):

        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            #print("frames ", len(fnames))
            #print("number of sequences ", len(fnames) // num_of_frames)
            s_fnames = sorted(fnames)
            for idx, fname in enumerate(s_fnames):
                if not fname.endswith('jpg'):
                    s_fnames.pop(idx)

            counter = 1
            item = []
            #print(sorted(fnames))
            num_img = len(s_fnames)
            if num_img > num_of_frames:
                # print("num_img", num_img)
                # print("num_frames", num_of_frames)
                # print("root", root)
                sample_size = 2 * (num_img // num_of_frames)
                #print(sample_size)
                for i in range(sample_size):
                    start = random.randrange(0, num_img - num_of_frames, 1)
                    item = []
                    append_ins = True
                    for j in range(num_of_frames):
                        if (is_image_file(os.path.join(root, s_fnames[start + j]))):
                            path = os.path.join(root, s_fnames[start + j])
                            path_ins = os.path.join(root.replace("JPEGImages", "Annotations"), s_fnames[start + j].replace("jpg", "png"))
                            #print(path)
                            #print(path_ins)
                            if not os.path.exists(path_ins):
                                print("this file does not exist: ", path_ins)
                            if not is_valid_file(path_ins):
                                print("error  here", path_ins)
                            if is_valid_file(path) and os.path.exists(path_ins):
                                temp = path, path_ins, class_index
                                item.append(temp)
                            else:
                                append_ins = False
                                break
                        else:
                            print("This file is not an image:", os.path.join(root, s_fnames[start + j]))

                    if append_ins:
                        instances.append(item)






            # for fname in s_fnames:
            #     if( is_image_file(os.path.join(root, fname))):
            #
            #         if ((counter) % num_of_frames == 0):
            #             path = os.path.join(root, fname)
            #             if is_valid_file(path):
            #                 temp = path, class_index
            #                 item.append(temp)
            #                 instances.append(item)
            #                 item = []
            #
            #         else:
            #             path = os.path.join(root, fname)
            #             if is_valid_file(path):
            #                 temp = path, class_index
            #                 item.append(temp)
            #
            #         counter +=1
            #     else:
            #         print("This file is not an image:", os.path.join(root, fname))
            #
    return instances


class DatasetFolderYVos(VisionDataset):
    """A generic data loader where the samples are arranged in this way: ::
        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext
        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext
    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        is_valid_file (callable, optional): A function that takes path of a file
            and check if the file is a valid file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.
     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(
            self,
            root: str,
            num_of_frames: int,
            loader: Callable[[str], Any],
            extensions: Optional[Tuple[str, ...]] = None,
            transform: Optional[Callable] = None,
            transform_masks: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> None:
        super(DatasetFolderYVos, self).__init__(root, transform=transform,
                                            target_transform=target_transform)
        self.num_of_frames = num_of_frames


        self.transform_masks = transform_masks


        # classes_instances, class_to_idx_instances = self._find_classes(os.path.join(self.root,"instances"))
        # samples_instances = make_dataset(os.path.join(self.root,"instances"), self.num_of_frames, class_to_idx_instances, extensions, is_valid_file)
        classes_imgs, class_to_idx_imgs = self._find_classes(os.path.join(self.root,"JPEGImages"))
        samples_imgs = make_dataset(os.path.join(self.root,"JPEGImages"), self.num_of_frames, class_to_idx_imgs, extensions, is_valid_file)
        # classes_instances, class_to_idx_instances = self._find_classes(os.path.join(self.root,"instances"))
        # samples_instances = make_dataset(os.path.join(self.root,"instances"), self.num_of_frames, class_to_idx_instances, extensions, is_valid_file)
        if len(samples_imgs) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(os.path.join(self.root,"images"))
            if extensions is not None:
                msg += "Supported extensions are: {}".format(",".join(extensions))
            raise RuntimeError(msg)
        # if len(samples_instances) == 0:
        #     msg = "Found 0 files in subfolders of: {}\n".format(os.path.join(self.root,"instances"))
        #     if extensions is not None:
        #         msg += "Supported extensions are: {}".format(",".join(extensions))
        #     raise RuntimeError(msg)
        
        self.loader = loader
        self.extensions = extensions

        self.classes_imgs = classes_imgs
        self.class_to_idx_imgs = class_to_idx_imgs
        self.samples_imgs = samples_imgs
        self.targets_imgs = [s[1] for s in samples_imgs]
        # self.classes_instances = classes_instances
        # self.class_to_idx_instances = class_to_idx_instances
        # self.samples_instances = samples_instances
        # self.targets_instances = [s[1] for s in samples_instances]

    def _find_classes(self, dir: str) -> Tuple[List[str], Dict[str, int]]:
        """
        Finds the class folders in a dataset.
        Args:
            dir (string): Root directory path.
        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.
        Ensures:
            No class is a subdirectory of another.
        """
        
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        imgs = []
        num_of_classes = 0
        unique_a = []

        # print("new item")

        start = random.randrange(0, self.num_of_frames, 1)
        path, path_ins, target = self.samples_imgs[index][start]
        f = open(path_ins, "rb")
        img = Image.open(f)
        masks = np.array(img)
        unique = np.unique(masks)
        ids = np.setdiff1d(unique, [0])
        if ids.size == 0:
            print("List is empty, only background")
            ids = [20000]

        id_selected = random.choice(ids)
        seq_of_class = []
        for i in range(self.num_of_frames):

            path, path_ins, target = self.samples_imgs[index][i]
            # print("path", path)

            sample = self.loader(path)

            f = open(path_ins, "rb")
            img = Image.open(f)
            masks = np.array(img)
            unique = np.unique(masks)

            # print("uni", unique)

            id_mask = (masks == id_selected).astype(float)  # * (i+1)
            # id_mask = np.expand_dims(id_mask, axis=0)

            # print("class", id)
            # print("unique2", unique2)
            PIL_image = Image.fromarray(id_mask)
            if self.transform_masks is not None:
                pil_img = self.transform_masks(PIL_image)
                seq_of_class.append(pil_img)

            if self.transform is not None:
                sample = self.transform(sample)
                imgs.append(sample)

            if self.target_transform is not None:
                target = self.target_transform(target)

        # print("imgs", len(imgs))

        rgbimgs = torch.stack([imgs[t] for t in range(self.num_of_frames)], dim=0)
        mask_single_class = torch.stack([seq_of_class[t] for t in range(self.num_of_frames)], dim=0)

        return torch.cat([rgbimgs, mask_single_class], dim=1), target

    def __len__(self) -> int:
        return len(self.samples_imgs)


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


# TODO: specify the return type
def accimage_loader(path: str) -> Any:
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path: str) -> Any:
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)




class ImageFolderYVos(DatasetFolderYVos):
    """A generic data loader where the images are arranged in this way: ::
        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png
        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)
     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(
            self,
            root: str,
            num_of_frames: int,
            transform: Optional[Callable] = None,
            transform_masks: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            img_ex: Optional[Callable] = None,
            loader: Callable[[str], Any] = default_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        super(ImageFolderYVos, self).__init__(root, num_of_frames, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                          transform=transform,
                                          transform_masks=transform_masks,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file)

        self.imgs = self.samples_imgs
        self.img_clases = self.classes_imgs
        self.img_class_to_idx = self.class_to_idx_imgs

        # self.imgs = self.samples
        # self.img_clases = self.classes
        # self.img_class_to_idx = self.class_to_idx

    # def __len__(self) -> int:
    #
    #   for classes in self.img_class_to_idx:
    #      return len(classes[1])

    # def _imgs_per_class(self) -> Tuple[Any, Any]:
    #
    #    for classes in self.img_class_to_idx:
    #       return len(classes[1])


#  train_transforms = transforms.Compose(
#         [transforms.Resize(255),
#          transforms.CenterCrop(224),
#          transforms.ToTensor(),
#          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

#
# def ucf_to_npy():
#     DATA_DIR = os.path.join("/globalwork/datasets/youtube-vos",
#                               "train")
#
#     train_transforms = transforms.Compose(
#         [transforms.Resize(256),
#          transforms.CenterCrop([256, 400]),
#          transforms.ToTensor()
#         ])
#
#     test_transforms = transforms.Compose(
#         [transforms.Resize(256, interpolation=Image.NEAREST),
#          transforms.CenterCrop([256, 400]),
#          transforms.ToTensor()
#         ])
#
#     dataset = ImageFolderYVos(DATA_DIR, 11, transform=train_transforms, transform_masks=test_transforms)
#
#
#
#
#
#     #torch.manual_seed(0)
#     dataset_train, dataset_val = torch.utils.data.random_split(dataset, [11000, 2372])
#
#     train_data_loader = torch.utils.data.DataLoader(dataset_train, batch_size=1,
#                                                     shuffle=False, num_workers=5 * max(1, 1), drop_last=True)
#
#     val_data_loader = torch.utils.data.DataLoader(dataset_val, batch_size=1,
#                                                     shuffle=False, num_workers=5 * max(1, 1), drop_last=True)
#
#
#
#     train_size = len(train_data_loader) * 1
#     train_size = len(train_data_loader) * 1
#     blini = True
#     print("blerttest", )
#
#     for v in val_data_loader:
#
#
#         print("blerttest", v[1])
#
#     # print(dataset.imgs[42][14])
#     # path = os.path.join("/globalwork/datasets/KITTI_MOTS",
#     #                           "train/instances/0000/000130.png")
#     # img = pil_loader(path)
#
#
#     print("train_data")
#     print(train_size)
#     c=0
#     for frames in train_data_loader:
#         print("frames_size", frames[0].size())
#         c+=1
#
#         for f in range(frames[0].size()[1]):
#             imgblin.imsave('testimg/name' + str(c) + '_' + str(1) +  '_' + str(f)  + '.png', np.float32(frames[0][0,f,0:3,:,:].permute(1, 2, 0) ) )
#             imgblin.imsave('testimg/name' + str(c) + '_' + str(1) + '_' + str(f)  + '_1.png', np.float32(frames[0][0, f, 3:4, :, :].repeat([3, 1, 1]).permute(1, 2, 0)))
#
#
#                     #print("Folder", frames[1])
#         #blini = False
#     # num_imgs_per_class = [0] * len(dataset.img_clases)
#     # print("djhvdkvd", len(dataset.img_clases))
#     # for i, classes in enumerate(dataset.img_class_to_idx):
#     #     print("i", i)
#     #     print("classes", classes)
#     #     for img in dataset.imgs:
#     #         if img[1] == i:
#     #             num_imgs_per_class[i] += 1
#
#     # print(num_imgs_per_class)
#
#
#
#
#     return dataset
#
# ucf = ucf_to_npy()