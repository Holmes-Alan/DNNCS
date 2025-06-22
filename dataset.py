import torch.utils.data as data
import torch
import numpy as np
import os
from os import listdir
from os.path import join
from PIL import Image, ImageOps
import random
from random import randrange
from torchvision import transforms
from video_util import frame_utils
import skimage

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

def is_flow_file(filename):
    return any(filename.endswith(extension) for extension in [".flo"])


def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    # y, _, _ = img.split()
    return img


def modcrop(im, modulo):
    (h, w) = im.size
    new_h = h//modulo*modulo
    new_w = w//modulo*modulo
    ih = h - new_h
    iw = w - new_w
    ims = im.crop((0, 0, h - ih, w - iw))
    return ims

def rescale_img(img_in, scale):
    size_in = img_in.size
    new_size_in = tuple([int(x * scale) for x in size_in])
    img_in = img_in.resize(new_size_in, resample=Image.BICUBIC)
    return img_in

def rescale_mask(img_in, scale):
    size_in = img_in.size
    new_size_in = tuple([int(x * scale) for x in size_in])
    img_in = img_in.resize(new_size_in, resample=Image.NEAREST)
    return img_in


def get_patch(img_in1, img_in2, img_tar1, img_tar2, img_tar3, img_mask, patch_size, scale, ix=-1, iy=-1):
    (ih, iw) = img_in1.size

    patch_mult = scale  # if len(scale) > 1 else 1
    tp = patch_mult * patch_size
    ip = tp // scale

    if ix == -1:
        ix = random.randrange(0, iw - tp + 1)
    if iy == -1:
        iy = random.randrange(0, ih - tp + 1)

    (tx, ty) = (scale * ix, scale * iy)

    out_in1 = img_in1.crop((iy, ix, iy + ip, ix + ip))
    out_in2 = img_in2.crop((iy, ix, iy + ip, ix + ip))
    out_tar1 = img_tar1.crop((iy, ix, iy + ip, ix + ip))
    out_tar2 = img_tar2.crop((iy, ix, iy + ip, ix + ip))
    out_tar3 = img_tar3.crop((iy, ix, iy + ip, ix + ip))
    out_mask = img_mask.crop((iy, ix, iy + tp, ix + tp))

    return out_in1, out_in2, out_tar1, out_tar2, out_tar3, out_mask, ix, iy, ip


def get_patch_v2(img_in1, img_in2, img_tar1, img_tar2, img_tar3, patch_size, scale, ix=-1, iy=-1):
    (ih, iw) = img_in1.size

    patch_mult = scale  # if len(scale) > 1 else 1
    tp = patch_mult * patch_size
    ip = tp // scale

    if ix == -1:
        ix = random.randrange(0, iw - tp + 1)
    if iy == -1:
        iy = random.randrange(0, ih - tp + 1)

    (tx, ty) = (scale * ix, scale * iy)

    out_in1 = img_in1.crop((iy, ix, iy + ip, ix + ip))
    out_in2 = img_in2.crop((iy, ix, iy + ip, ix + ip))
    out_tar1 = img_tar1.crop((iy, ix, iy + ip, ix + ip))
    out_tar2 = img_tar2.crop((iy, ix, iy + ip, ix + ip))
    out_tar3 = img_tar3.crop((iy, ix, iy + ip, ix + ip))

    #info_patch = {
    #    'ix': ix, 'iy': iy, 'ip': ip, 'tx': tx, 'ty': ty, 'tp': tp}

    return out_in1, out_in2, out_tar1, out_tar2, out_tar3


def get_patch_flow(img_in1, img_in2, img_tar1, img_tar2, img_tar3, img_mask, flow1, flow2, patch_size, scale, ix=-1, iy=-1):
    (ih, iw) = img_in1.size

    patch_mult = scale  # if len(scale) > 1 else 1
    tp = patch_mult * patch_size
    ip = tp // scale

    if ix == -1:
        ix = random.randrange(0, iw - tp + 1)
    if iy == -1:
        iy = random.randrange(0, ih - tp + 1)

    (tx, ty) = (scale * ix, scale * iy)

    out_in1 = img_in1.crop((iy, ix, iy + ip, ix + ip))
    out_in2 = img_in2.crop((iy, ix, iy + ip, ix + ip))
    out_tar1 = img_tar1.crop((iy, ix, iy + ip, ix + ip))
    out_tar2 = img_tar2.crop((iy, ix, iy + ip, ix + ip))
    out_tar3 = img_tar3.crop((iy, ix, iy + ip, ix + ip))
    out_mask = img_mask.crop((iy, ix, iy + tp, ix + tp))

    out_flow1 = flow1[:, iy:iy+ip, ix:ix+ip]
    out_flow2 = flow2[:, iy:iy+ip, ix:ix+ip]

    return out_in1, out_in2, out_tar1, out_tar2, out_tar3, out_mask, out_flow1, out_flow2


def augment(img_in1, img_in2, img_tar1, img_tar2, img_tar3, img_mask=None, flip_h=True, rot=True):
    info_aug = {'flip_h': False, 'flip_v': False, 'trans': False}

    if random.random() < 0.5 and flip_h:
        img_in1 = ImageOps.flip(img_in1)
        img_in2 = ImageOps.flip(img_in2)
        img_tar1 = ImageOps.flip(img_tar1)
        img_tar2 = ImageOps.flip(img_tar2)
        img_tar3 = ImageOps.flip(img_tar3)
        if img_mask is not None:
            img_mask = ImageOps.flip(img_mask)
        #img_bic = ImageOps.flip(img_bic)
        info_aug['flip_h'] = True

    if rot:
        if random.random() < 0.5:
            img_in1 = ImageOps.mirror(img_in1)
            img_in2 = ImageOps.mirror(img_in2)
            img_tar1 = ImageOps.mirror(img_tar1)
            img_tar2 = ImageOps.mirror(img_tar2)
            img_tar3 = ImageOps.mirror(img_tar3)
            if img_mask is not None:
                img_mask = ImageOps.mirror(img_mask)
            info_aug['flip_v'] = True
        if random.random() < 0.5:
            img_in1 = img_in1.rotate(90)
            img_in2 = img_in2.rotate(90)
            img_tar1 = img_tar1.rotate(90)
            img_tar2 = img_tar2.rotate(90)
            img_tar3 = img_tar3.rotate(90)
            if img_mask is not None:
                img_mask = img_mask.rotate(90)
            info_aug['trans'] = True

    return img_in1, img_in2, img_tar1, img_tar2, img_tar3, img_mask, info_aug


def augment_flow(img_in1, img_in2, img_tar1, img_tar2, img_tar3, flow1, flow2, img_mask=None, flip_h=True, rot=True):
    info_aug = {'flip_h': False, 'flip_v': False, 'trans': False}

    if random.random() < 0.5 and flip_h:
        img_in1 = ImageOps.flip(img_in1)
        img_in2 = ImageOps.flip(img_in2)
        img_tar1 = ImageOps.flip(img_tar1)
        img_tar2 = ImageOps.flip(img_tar2)
        img_tar3 = ImageOps.flip(img_tar3)
        flow1 = torch.flip(flow1, [1])
        flow2 = torch.flip(flow2, [1])
        if img_mask is not None:
            img_mask = ImageOps.flip(img_mask)
        #img_bic = ImageOps.flip(img_bic)
        info_aug['flip_h'] = True

    if rot:
        if random.random() < 0.5:
            img_in1 = ImageOps.mirror(img_in1)
            img_in2 = ImageOps.mirror(img_in2)
            img_tar1 = ImageOps.mirror(img_tar1)
            img_tar2 = ImageOps.mirror(img_tar2)
            img_tar3 = ImageOps.mirror(img_tar3)
            flow1 = torch.flip(flow1, [2])
            flow2 = torch.flip(flow2, [2])
            if img_mask is not None:
                img_mask = ImageOps.mirror(img_mask)
            info_aug['flip_v'] = True
        if random.random() < 0.5:
            img_in1 = img_in1.rotate(90)
            img_in2 = img_in2.rotate(90)
            img_tar1 = img_tar1.rotate(90)
            img_tar2 = img_tar2.rotate(90)
            img_tar3 = img_tar3.rotate(90)
            flow1 = torch.rot90(flow1, 1, [1, 2])
            flow2 = torch.rot90(flow2, 1, [1, 2])
            if img_mask is not None:
                img_mask = img_mask.rotate(90)
            info_aug['trans'] = True

    return img_in1, img_in2, img_tar1, img_tar2, img_tar3, flow1, flow2, img_mask, info_aug




def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs, indexing='ij'), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    ret = ret.numpy()
    return ret

class DatasetFromFolder_v3(data.Dataset):
    def __init__(self, data_dir, patch_size, upscale_factor, data_augmentation):
        super(DatasetFromFolder_v3, self).__init__()
        sub_folder = ['bahamas/1696', 'bahamas/6784', 'bahamas/27136', 'galveston/3397', 'galveston/13000', 'galveston/54000']
        # sub_folder = ['bahamas/1696']
        self.hr_images = []
        self.lr_images = []
        for t, folder in enumerate(sub_folder):
            for k in range(2):
                scale_hr = '_P' + str(k+1)
                scale_lr = '_P' + str(k)
                HR_subfolder = join(data_dir, folder + scale_hr)
                LR_subfolder = join(data_dir, folder + scale_lr)
                self.hr_image_filenames = sorted([join(HR_subfolder, x) for x in listdir(HR_subfolder) if is_image_file(x)])
                self.lr_image_filenames = sorted([join(LR_subfolder, x) for x in listdir(LR_subfolder) if is_image_file(x)])
                # len_file = len(self.lr_image_filenames) // 2
                for i in range(len(self.lr_image_filenames) - 1):
                    lr_file_1 = self.lr_image_filenames[i]
                    lr_file_2 = self.lr_image_filenames[i+1]
                    hr_file_1 = self.hr_image_filenames[2*i]
                    hr_file_2 = self.hr_image_filenames[2*i+1]
                    hr_file_3 = self.hr_image_filenames[2*i+2]
                    if t < 3:
                        mask = '/data/shallow_water/Train_v3/bahamas/mask.jpg'
                    else:
                        mask = '/data/shallow_water/Train_v3/galveston/mask.jpg'
                    self.lr_images.append([lr_file_1, lr_file_2, mask])
                    self.hr_images.append([hr_file_1, hr_file_2, hr_file_3])
        # self.hr_image_filenames = sorted(self.hr_image_filenames)
        # self.lr_image_filenames = sorted(self.lr_image_filenames)
        self.patch_size = patch_size
        self.upscale_factor = upscale_factor
        self.data_augmentation = data_augmentation

    def __getitem__(self, index):

        target1 = load_img(self.hr_images[index][0])
        target1 = modcrop(target1, 4)
        target2 = load_img(self.hr_images[index][1])
        target2 = modcrop(target2, 4)
        target3 = load_img(self.hr_images[index][2])
        target3 = modcrop(target3, 4)
        input1 = load_img(self.lr_images[index][0])
        input1 = modcrop(input1, 4)
        input2 = load_img(self.lr_images[index][1])
        input2 = modcrop(input2, 4)
        mask = load_img(self.lr_images[index][2])
        mask = modcrop(mask, 4)

        input1, input2, target1, target2, target3, mask = get_patch(input1, input2, target1, target2, target3, mask, self.patch_size, self.upscale_factor)

        if self.data_augmentation:
            input1, input2, target1, target2, target3, mask, _ = augment(input1, input2, target1, target2, target3, mask)

        input1 = transforms.ToTensor()(input1)
        input2 = transforms.ToTensor()(input2)
        target1 = transforms.ToTensor()(target1)
        target2 = transforms.ToTensor()(target2)
        target3 = transforms.ToTensor()(target3)
        mask = transforms.ToTensor()(mask)

        if random.random() < 0.5:
            a = input1
            input1 = input2
            input2 = a
            a = target1
            target1 = target3
            target3 = a

        return input1, input2, target1, target2, target3, mask

    def __len__(self):
        return len(self.lr_images)

class DatasetFromFolder_v5(data.Dataset):
    def __init__(self, data_dir, patch_size, upscale_factor, data_augmentation):
        super(DatasetFromFolder_v5, self).__init__()
        sub_folder = ['bahamas/1696', 'bahamas/6784', 'bahamas/27136']
        # sub_folder = ['galveston/3397', 'galveston/13000', 'galveston/54000']
        # sub_folder = ['bahamas/1696']
        self.hr_images = []
        self.lr_images = []
        for t in range(2):
            LR_folder = sub_folder[t]
            HR_folder = sub_folder[t+1]
            for k in range(0, 3, 1):
                scale_hr = '_P' + str(k)
                scale_lr = '_P' + str(k)
                HR_subfolder = join(data_dir, HR_folder + scale_hr)
                LR_subfolder = join(data_dir, LR_folder + scale_lr)
                self.hr_image_filenames = sorted([join(HR_subfolder, x) for x in listdir(HR_subfolder) if is_image_file(x)])
                self.lr_image_filenames = sorted([join(LR_subfolder, x) for x in listdir(LR_subfolder) if is_image_file(x)])
                # len_file = len(self.lr_image_filenames) // 2
                for i in range(len(self.lr_image_filenames) - 1):
                    lr_file_1 = self.lr_image_filenames[i]
                    lr_file_2 = self.lr_image_filenames[i+1]
                    hr_file_1 = self.hr_image_filenames[2*i]
                    hr_file_2 = self.hr_image_filenames[2*i+1]
                    hr_file_3 = self.hr_image_filenames[2*i+2]
                    mask = '/data/shallow_water/Train_v3/bahamas/mask.jpg'
                    # mask = '/data/shallow_water/Train_v3/galveston/mask.jpg'
                    self.lr_images.append([lr_file_1, lr_file_2, mask])
                    self.hr_images.append([hr_file_1, hr_file_2, hr_file_3])

        sub_folder = ['galveston/3397', 'galveston/13000', 'galveston/54000']
        for t in range(2):
            LR_folder = sub_folder[t]
            HR_folder = sub_folder[t+1]
            for k in range(3):
                scale_hr = '_P' + str(k)
                scale_lr = '_P' + str(k)
                HR_subfolder = join(data_dir, HR_folder + scale_hr)
                LR_subfolder = join(data_dir, LR_folder + scale_lr)
                self.hr_image_filenames = sorted([join(HR_subfolder, x) for x in listdir(HR_subfolder) if is_image_file(x)])
                self.lr_image_filenames = sorted([join(LR_subfolder, x) for x in listdir(LR_subfolder) if is_image_file(x)])
                # len_file = len(self.lr_image_filenames) // 2
                for i in range(len(self.lr_image_filenames) - 1):
                    lr_file_1 = self.lr_image_filenames[i]
                    lr_file_2 = self.lr_image_filenames[i+1]
                    hr_file_1 = self.hr_image_filenames[2*i]
                    hr_file_2 = self.hr_image_filenames[2*i+1]
                    hr_file_3 = self.hr_image_filenames[2*i+2]
                    # mask = '/data/shallow_water/Train_v3/bahamas/mask.jpg'
                    mask = '/data/shallow_water/Train_v3/galveston/mask.jpg'
                    self.lr_images.append([lr_file_1, lr_file_2, mask])
                    self.hr_images.append([hr_file_1, hr_file_2, hr_file_3])

        self.patch_size = patch_size
        self.upscale_factor = upscale_factor
        self.data_augmentation = data_augmentation

    def __getitem__(self, index):

        target1 = load_img(self.hr_images[index][0])
        target1 = modcrop(target1, 4)
        target2 = load_img(self.hr_images[index][1])
        target2 = modcrop(target2, 4)
        target3 = load_img(self.hr_images[index][2])
        target3 = modcrop(target3, 4)
        input1 = load_img(self.lr_images[index][0])
        input1 = modcrop(input1, 4)
        input2 = load_img(self.lr_images[index][1])
        input2 = modcrop(input2, 4)
        mask = load_img(self.lr_images[index][2])
        mask = modcrop(mask, 4)

        target1 = rescale_img(target1, 0.5)
        target2 = rescale_img(target2, 0.5)
        target3 = rescale_img(target3, 0.5)
        input1 = rescale_img(input1, 0.5)
        input2 = rescale_img(input2, 0.5)
        mask = rescale_img(mask, 0.5)
        mask_array = np.array(mask)
        coord = make_coord(mask_array.shape[:-1], flatten=False)

        input1, input2, target1, target2, target3, mask, ix, iy, ip = get_patch(input1, input2, target1, target2, target3, mask, self.patch_size, self.upscale_factor)

        coord = coord[ix:ix+ip, iy:iy+ip, :]

        if self.data_augmentation:
            input1, input2, target1, target2, target3, mask, info_aug = augment(input1, input2, target1, target2, target3, mask)

        if info_aug['flip_h'] is True:
            coord = np.flip(coord, axis=0)
        if info_aug['flip_v'] is True:
            coord = np.flip(coord, axis=1)
        if info_aug['trans'] is True:
            coord = np.rot90(coord)

        input1 = transforms.ToTensor()(input1)
        input2 = transforms.ToTensor()(input2)
        target1 = transforms.ToTensor()(target1)
        target2 = transforms.ToTensor()(target2)
        target3 = transforms.ToTensor()(target3)
        mask = transforms.ToTensor()(mask)
        coord = transforms.ToTensor()(coord.copy())

        if random.random() < 0.5:
            a = input1
            input1 = input2
            input2 = a
            a = target1
            target1 = target3
            target3 = a

        return input1, input2, target1, target2, target3, mask, coord

    def __len__(self):
        return len(self.lr_images)
    

class DatasetFromFolderEval_v5(data.Dataset):
    def __init__(self, data_dir, patch_size, upscale_factor):
        super(DatasetFromFolderEval_v5, self).__init__()
        sub_folder = ['bahamas/1696', 'bahamas/6784', 'bahamas/27136']
        # sub_folder = ['galveston/3397', 'galveston/13000', 'galveston/54000']
        self.hr_images = []
        self.lr_images = []
        for t in range(2):
            LR_folder = sub_folder[t]
            HR_folder = sub_folder[t+1]
            for k in range(0, 3, 1):
                scale_hr = '_P' + str(k)
                scale_lr = '_P' + str(k)
                HR_subfolder = join(data_dir, HR_folder + scale_hr)
                LR_subfolder = join(data_dir, LR_folder + scale_lr)
                self.hr_image_filenames = sorted([join(HR_subfolder, x) for x in listdir(HR_subfolder) if is_image_file(x)])
                self.lr_image_filenames = sorted([join(LR_subfolder, x) for x in listdir(LR_subfolder) if is_image_file(x)])
                # len_file = len(self.lr_image_filenames) // 32
                len_file = 20
                for i in range(len_file - 1):
                    lr_file_1 = self.lr_image_filenames[i]
                    lr_file_2 = self.lr_image_filenames[i+1]
                    hr_file_1 = self.hr_image_filenames[2*i]
                    hr_file_2 = self.hr_image_filenames[2*i+1]
                    hr_file_3 = self.hr_image_filenames[2*i+2]
                    mask = '/data/shallow_water/Train_v3/bahamas/mask.jpg'
                    # mask = '/data/shallow_water/Train_v3/galveston/mask.jpg'
                    self.lr_images.append([lr_file_1, lr_file_2, mask])
                    self.hr_images.append([hr_file_1, hr_file_2, hr_file_3])

        sub_folder = ['galveston/3397', 'galveston/13000', 'galveston/54000']
        for t in range(2):
            LR_folder = sub_folder[t]
            HR_folder = sub_folder[t+1]
            for k in range(3):
                scale_hr = '_P' + str(k)
                scale_lr = '_P' + str(k)
                HR_subfolder = join(data_dir, HR_folder + scale_hr)
                LR_subfolder = join(data_dir, LR_folder + scale_lr)
                self.hr_image_filenames = sorted([join(HR_subfolder, x) for x in listdir(HR_subfolder) if is_image_file(x)])
                self.lr_image_filenames = sorted([join(LR_subfolder, x) for x in listdir(LR_subfolder) if is_image_file(x)])
                # len_file = len(self.lr_image_filenames) // 32
                len_file = 20
                for i in range(len_file - 1):
                    lr_file_1 = self.lr_image_filenames[i]
                    lr_file_2 = self.lr_image_filenames[i+1]
                    hr_file_1 = self.hr_image_filenames[2*i]
                    hr_file_2 = self.hr_image_filenames[2*i+1]
                    hr_file_3 = self.hr_image_filenames[2*i+2]
                    # mask = '/data/shallow_water/Train_v3/bahamas/mask.jpg'
                    mask = '/data/shallow_water/Train_v3/galveston/mask.jpg'
                    self.lr_images.append([lr_file_1, lr_file_2, mask])
                    self.hr_images.append([hr_file_1, hr_file_2, hr_file_3])

        self.patch_size = patch_size
        self.upscale_factor = upscale_factor

    def __getitem__(self, index):

        target1 = load_img(self.hr_images[index][0])
        target1 = modcrop(target1, 4)
        target2 = load_img(self.hr_images[index][1])
        target2 = modcrop(target2, 4)
        target3 = load_img(self.hr_images[index][2])
        target3 = modcrop(target3, 4)
        input1 = load_img(self.lr_images[index][0])
        input1 = modcrop(input1, 4)
        input2 = load_img(self.lr_images[index][1])
        input2 = modcrop(input2, 4)
        mask = load_img(self.lr_images[index][2])
        mask = modcrop(mask, 4)

        target1 = rescale_img(target1, 0.5)
        target2 = rescale_img(target2, 0.5)
        target3 = rescale_img(target3, 0.5)
        input1 = rescale_img(input1, 0.5)
        input2 = rescale_img(input2, 0.5)
        mask = rescale_img(mask, 0.5)
        mask_array = np.array(mask)
        coord = make_coord(mask_array.shape[:-1], flatten=False)
        
        input1 = transforms.ToTensor()(input1)
        input2 = transforms.ToTensor()(input2)
        target1 = transforms.ToTensor()(target1)
        target2 = transforms.ToTensor()(target2)
        target3 = transforms.ToTensor()(target3)
        mask = transforms.ToTensor()(mask)
        coord = transforms.ToTensor()(coord.copy())

        return input1, input2, target1, target2, target3, mask, coord

    def __len__(self):
        return len(self.lr_images)
    


class DatasetFromFolderTest_demo(data.Dataset):
    def __init__(self, lr_folder, hr_folder, data_dir, patch_size, upscale_factor, file_idx):
        super(DatasetFromFolderTest_demo, self).__init__()
        self.hr_images = []
        self.lr_images = []

        for k in range(1, 3, 1):
            scale = '_P' + str(k)
            name_idf = lr_folder.split('/')[0]
            HR_subfolder = join(data_dir, hr_folder + scale)
            LR_subfolder = join(data_dir, lr_folder + scale)
            self.hr_image_filenames = sorted([join(HR_subfolder, x) for x in listdir(HR_subfolder) if is_image_file(x)])
            self.lr_image_filenames = sorted([join(LR_subfolder, x) for x in listdir(LR_subfolder) if is_image_file(x)])
            len_file = len(self.lr_image_filenames)
            # len_file = 20
            if name_idf == 'bahamas':
                step = (k + file_idx + 1) * 20
            else:
                step = (k + file_idx + 1) * 10
            for i in range(0, len_file, step):
                lr_file_1 = self.lr_image_filenames[i]
                lr_file_2 = self.lr_image_filenames[i+1]
                hr_file_1 = self.hr_image_filenames[2*i]
                hr_file_2 = self.hr_image_filenames[2*i+1]
                hr_file_3 = self.hr_image_filenames[2*i+2]
                if name_idf == 'bahamas':
                    mask = '/data/shallow_water/Train_v3/bahamas/mask.jpg'
                else:
                    mask = '/data/shallow_water/Train_v3/galveston/mask.jpg'
                # HR_name = 'P' + str(k)
                a = hr_file_1.split('/')[-2]
                b = hr_file_1.split('/')[-1]
                HR_name = a + b[3:]
                # LR_name = 'P' + str(k)
                self.lr_images.append([lr_file_1, lr_file_2, mask])
                self.hr_images.append([hr_file_1, hr_file_2, hr_file_3, HR_name])
        # self.hr_image_filenames = sorted(self.hr_image_filenames)
        # self.lr_image_filenames = sorted(self.lr_image_filenames)
        self.patch_size = patch_size
        self.upscale_factor = upscale_factor

    def __getitem__(self, index):

        target1 = load_img(self.hr_images[index][0])
        target1 = modcrop(target1, 4)
        target2 = load_img(self.hr_images[index][1])
        target2 = modcrop(target2, 4)
        target3 = load_img(self.hr_images[index][2])
        target3 = modcrop(target3, 4)
        input1 = load_img(self.lr_images[index][0])
        input1 = modcrop(input1, 4)
        input2 = load_img(self.lr_images[index][1])
        input2 = modcrop(input2, 4)
        mask = load_img(self.lr_images[index][2])
        mask = modcrop(mask, 4)

        target1 = rescale_img(target1, 0.5)
        target2 = rescale_img(target2, 0.5)
        target3 = rescale_img(target3, 0.5)
        input1 = rescale_img(input1, 0.5)
        input2 = rescale_img(input2, 0.5)
        mask = rescale_img(mask, 0.5)
        mask_array = np.array(mask)
        coord = make_coord(mask_array.shape[:-1], flatten=False)

        a = self.hr_images[index][0].split('/')
        b = self.hr_images[index][1].split('/')
        file_name1 = a[-2] + '_' +  a[-1].split('_')[-1]
        file_name2 = b[-2] + '_' + b[-1].split('_')[-1]

        input1 = transforms.ToTensor()(input1)
        input2 = transforms.ToTensor()(input2)
        target1 = transforms.ToTensor()(target1)
        target2 = transforms.ToTensor()(target2)
        target3 = transforms.ToTensor()(target3)
        mask = transforms.ToTensor()(mask)
        coord = transforms.ToTensor()(coord.copy())

        return input1, input2, target1, target2, target3, mask, coord, file_name1, file_name2

    def __len__(self):
        return len(self.lr_images)