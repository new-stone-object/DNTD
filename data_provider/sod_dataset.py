import os
import numpy as np
import cv2
from torch.utils import data
import torch
from data_provider.img_utils import random_scale_pair_image
from data_provider.img_utils import square_crop
from data_provider.img_utils import gen_edge_image
from typing import Tuple
import random


class SaliencyDataSet(data.Dataset):

    def __init__(self, root_dir:str, list_path:str, ignore_value:float = 255.0, \
                 crop_size:Tuple[int,int] = (321, 321),
                 img_mean:np.ndarray = np.array([ 0.485, 0.456, 0.406]), \
                 img_std:np.ndarray = np.array([0.229, 0.224, 0.225]), \
                 is_random_flip:bool = False, is_scale:bool = False,is_use_edge=False,
                 is_random_brightness = False) -> None:
        img_ids = [i_id.strip() for i_id in open(list_path) if i_id.strip() != '']
        self.files = []
        self.img_mean = img_mean.reshape(1,1,3)
        self.img_std = img_std.reshape(1,1,3)
        for img_id in img_ids:
            img_name, gt_name = img_id.split()
            img_file = os.path.join(root_dir, img_name)
            label_file = os.path.join(root_dir, gt_name)
            self.files.append({
                "img": img_file,
                "label": label_file,
            })
        self.is_random_flip:bool = is_random_flip
        self.is_scale:bool = is_scale
        self.ignore_value:float = ignore_value
        self.pad_value = (0 - img_mean) / img_std
        if crop_size is not None and crop_size != (0,0):
            self.crop_height,self.crop_width = crop_size
        else:
            self.crop_height, self.crop_width = None,None
        self.is_random_brightness = is_random_brightness

    def __len__(self):
        return len(self.files)

    def __getitem__( self, index: int):

        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        gray_label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)
        if self.is_random_brightness:
            img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            m = np.max(img_hsv[..., -1])
            m = list(range(255 - m))
            l = np.min(img_hsv[..., -1])
            m += [-i for i in range(l + 1)]
            m = np.random.choice(m)
            img_hsv[..., -1] = img_hsv[..., -1] + int(m)
            image = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)


        if self.is_scale:
            image,gray_label = random_scale_pair_image(image, gray_label)

        if self.crop_width is not None and self.crop_height is not None:
            image, gray_label = square_crop(image, gray_label,need_size=self.crop_height)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        filename = os.path.basename(datafiles["img"])
        image = image.astype(np.float32)
        image /= 255.0
        image -= self.img_mean
        image /= self.img_std

        label = gray_label.astype(np.float32)
        label /= 255


        image = image.transpose((2, 0, 1))
        if self.is_random_flip:
            if random.randint(0,1) == 1:
                image = image[..., ::-1]
                label = label[..., ::-1]


        return {
            'image':image.copy(),
            'label':label.copy(),
            'filename':filename,
        }

    @torch.no_grad()
    def real_image(self, image:torch.Tensor, ignore_index:torch.Tensor = None) -> torch.Tensor:
        """
        recover the real image from tensor
        :param image:
        :param ignore_index: the bool type tensor with shape [N,H,W] or [N,C,H,W]
        :return: the real image value with shape [N,C,H,W]
        """
        # print(image.shape)
        if self.crop_width is not None and self.crop_height is not None:
            if ignore_index is not None:
                if ignore_index.dim() == 4:
                    ignore_index = ignore_index[:,0,:,:]
            else:
                ignore_index = (image == 0.0)
                ignore_index = ignore_index[:,0,:,:]
            for i in range(ignore_index.size(0)):
                image[i,:,ignore_index[i]] = -torch.from_numpy(self.img_mean).to(image.dtype).view(-1,1).expand_as(image[i,:,ignore_index[i]])
                image[i,:,ignore_index[i]] /= torch.from_numpy(self.img_std).to(image.dtype).view(-1,1).expand_as(image[i,:,ignore_index[i]])
        image *= torch.from_numpy(self.img_std).to(image.dtype).view(1, -1, 1, 1).float()
        image += torch.from_numpy(self.img_mean).to(image.dtype).view(1,-1,1,1).float()
        image *= 255.0
        image = image.byte()
        return image

    def real_label(self, label:torch.Tensor) -> torch.Tensor:
        """
        return the real image value with shape [C,H,W]
        :param label: the label shape must be [N,H,W] or [N,C,H,W]
        :return:
        """
        ignore_index = (label == self.ignore_value)
        label *= 255.0
        label[ignore_index] == 0.0
        if len(label.shape) == 3:
            label = label.unsqueeze(dim=1)
        if label.size(1) == 1:
            shape = list(label.shape)
            shape[1] = 3
            label = label.expand(shape)
        label = label.byte()
        return label


if __name__ == '__main__':
    dst = SaliencyDataSet("../dataset/SOD", '../dataset/SOD/train.lst')
    trainloader = data.DataLoader(dst,batch_size=1,shuffle=True)
    from torchvision.utils import make_grid
    import torch
    for i, data in enumerate(trainloader):
        print(type(data['image']))
        print(data['image'].shape)
        print(data['label'].shape)
        print(data['filename'])
        img = dst.real_image(data['image'].cpu())
        label = dst.real_label(data['label'].cpu())
        print(img.shape,label.shape)
        show_image = make_grid(torch.cat([img,label],dim=3),nrow=5)
        cv2.imshow('test', cv2.cvtColor(show_image.numpy().transpose((1,2,0)),cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)







