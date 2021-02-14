from networks import DNTD as Net
from torch.utils.data import DataLoader
import torch
from tqdm import trange
from typing import Callable
from helper.visual_helper import VisualHelper
from helper.model_saver import ModelSaver
from helper.optim import Optimizer
import torch.nn.functional as F
LOSS_FUNC = Callable[[torch.Tensor,torch.Tensor],torch.Tensor]
WEIGHT_INIT_FUNC = Callable[[Net],None]
__all__ = ['Trainer']


class Trainer(object):
    """
    the helper to train network model
    """
    def __init__(self,
                 dataloader: DataLoader,
                 epoch:int,
                 optim_create_func:Optimizer,
                 lr:float,
                 loss_function:LOSS_FUNC,
                 device: torch.device = torch.device('cpu'),
                 pretrained_model_path:str = None,
                 visual_helper:VisualHelper = None,
                 model_saver:ModelSaver = None,
                 weight_init_func:WEIGHT_INIT_FUNC = None,
                 drop_rate = 0,
                 *args,
                 **kwargs):
        super(Trainer,self).__init__(*args,**kwargs)
        cpu_device = torch.device('cpu')
        self.model:Net = Net(drop_rate=drop_rate).to(cpu_device)
        self.model.train()
        self.dataloader = dataloader
        if weight_init_func is not None:
            self.model.apply(weight_init_func)
        if pretrained_model_path is not None:
            ret = self.model.load_encoder_weight(pretrained_model_path)
            print(ret)

        self.epoch = epoch
        optim_create_func(self.model, lr)
        self.optim = optim_create_func
        self.device = device
        self.loss_func = loss_function
        self.visual_helper = visual_helper
        self.model_saver = model_saver
        self.start_epoch = 0
        self.model.to(device)

    def train(self):
        if self.visual_helper is not None:
            loss_list = list()

        for epoch in trange(self.start_epoch,self.epoch):

            for item in self.dataloader:
                imgs = item['image'].to(self.device)
                labels = item['label'].to(self.device)
                if imgs.shape[-2:] != labels.shape[-2:]:
                    del imgs
                    del labels
                    print(f"{item['filename']} size didn't match!")
                    continue
                preds = self.model(imgs)
                losses = 0
                if labels.dim() == 3:
                    labels = labels.unsqueeze(dim=1)

                if labels.shape[-2:] != preds.shape[-2:]:
                    label = F.interpolate(labels, size=preds.shape[-2:], mode='bilinear', align_corners=True)
                losses += self.loss_func(preds, label)

                losses = losses / self.optim.step_time_interval
                losses.backward()

                if self.visual_helper is not None:
                    loss_list.append(losses.item())
                    self.visual_helper.add_timer()
                    loss_value = sum(loss_list) / (len(loss_list) / self.optim.step_time_interval)
                    if self.visual_helper.is_catch_snapshot():
                        self.visual_helper(epoch,
                                           loss_value,
                                           dict(imgs=imgs.cpu().detach(),
                                                labels=labels.cpu().detach(),
                                                preds=preds.cpu().detach())
                                           )
                        loss_list.clear()
                self.optim.step()
                if self.model_saver is not None:
                    self.model_saver({
                        'model':self.model.state_dict(),
                        'optim':self.optim.state_dict(),
                        'saver':self.model_saver.state_dict(),
                        'start_epoch':epoch,
                    })
        if self.model_saver is not None:
            self.model_saver({
                        'model':self.model.state_dict(),
                        'optim':self.optim.state_dict(),
                        'saver':self.model_saver.state_dict(),
                        'start_epoch':epoch,
                    },isFinal=True)
        if self.visual_helper is not None:
            self.visual_helper.close()
