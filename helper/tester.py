from networks import DenseUNet as Net
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from helper.utils import mkdirs
from helper.utils import pathjoin
from typing import Callable
import os
import numpy as np
import cv2
import time
LABEL_TRANS_FUNC = Callable[[torch.Tensor],torch.Tensor]
__all__ = ['Tester']


class Tester(object):
    """
    the helper to train network model
    """
    def __init__(self,
                 dataloader: DataLoader,
                 img_save_path: str,
                 model_pth_path: str,
                 device:torch.device = torch.device('cpu'),
                 label_trans_func:LABEL_TRANS_FUNC = lambda x:x,
                 *args,
                 **kwargs) -> None:
        super(Tester,self).__init__(*args,**kwargs)
        self._dataloader = dataloader
        self._img_save_path = img_save_path
        mkdirs(img_save_path)
        self._device = device
        self._label_transform_func = label_trans_func
        model_dict = torch.load(model_pth_path,map_location='cpu')['model']
        self._model = Net()
        self._model.load_state_dict(model_dict)
        del model_dict
        self._model = self._model.to(device)
        self._model.eval()


    def test(self) -> None:
        dataset_len = len(self._dataloader.dataset)
        pbar = tqdm(total=dataset_len)
        with torch.no_grad():
            start_time = time.time()
            for item in self._dataloader:
                imgs = item['image'].to(self._device)
                size = imgs.shape[-2:]
                preds = self._model(imgs)
                if isinstance(preds,(list,tuple)):
                    preds = preds[0].cpu()
                else:
                    preds = preds.cpu()
                preds = self._label_transform_func(preds)
                for i in range(preds.shape[0]):
                    filename = os.path.splitext(item['filename'][i])[0]+'.png'
                    filepath = pathjoin(self._img_save_path,filename)
                    img = preds[i]
                    # if img.shape[-2:] != size:
                    #     img = cv2.resize(img,dsize=(size[1],size[0]),interpolation=cv2.INTER_LINEAR)
                    cv2.imwrite(filepath, img)
                    #pbar.update(1)
                    #pbar.set_description("Processing %s" % filename)
            print(f"{ dataset_len / (time.time()-start_time) } FPS")
