from config import Configuration
from helper.visual_helper import VisualHelper
from helper.visual_helper import ImgsType
from helper.model_saver import ModelSaver
from torch.nn import Module
from networks import DenseUNet as Net
from helper.optim import Optimizer
from torch.optim import Adam
import torch.nn as nn
from torch.utils.data import DataLoader
import torch
import os
from helper.trainer import Trainer
from helper.utils import pathjoin
from helper.utils import mkdirs
from helper.utils import pandas2markdown
import pandas as pd
from helper.tester import Tester
from helper.evaluator_tool import get_measure
from collections.abc import Sequence
from collections import OrderedDict
import time
from torchvision.utils import make_grid
import portalocker
from networks import get_upsampling_weight
from losses import structure_weighted_binary_cross_entropy_with_logits
import torch.nn.functional as F
from data_provider.sod_dataset import SaliencyDataSet


def train(config:Configuration) -> None:

    if config.DISABLE_TRAIN:
        return

    batch_size = config.BATCH_SIZE
    epoch = config.EPOCH
    lr = config.LEARNING_RATE
    crop_size = config.CROP_SIZE
    dataset = SaliencyDataSet(config.DATASET_TRAIN_ROOT_DIR, config.DATASET_TRAIN_LIST_PATH, \
                              crop_size=(crop_size,crop_size),is_scale=config.DATASET_IS_SCALE,
                              is_random_flip=True,ignore_value=255,is_random_brightness=config.RANDOM_BRIGHT)
    if batch_size > 1 and torch.cuda.device_count() > 0:
        torch.backends.cudnn.enable = True
        torch.backends.cudnn.benchmark = True
    trainloader = DataLoader(dataset, batch_size=batch_size, num_workers= batch_size if batch_size < 10 else 10, drop_last=True, shuffle=True, pin_memory=True)
    if config.USE_GPU is not None:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # weight initial function
    def weight_init( m:nn.Module):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias,0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
            m.eval()
            for name,p in m.named_parameters():
                p.requires_grad_(False)
        elif isinstance(m, nn.Dropout):
            m.eval()
        elif isinstance(m, nn.ConvTranspose2d):
            assert m.kernel_size[0] == m.kernel_size[1]
            initial_weight = get_upsampling_weight(
                m.in_channels, m.out_channels // m.groups, m.kernel_size[0])
            m.weight.data.copy_(initial_weight)
            nn.init.constant_(m.bias, 0)

    loss_func = structure_weighted_binary_cross_entropy_with_logits
    weight_init_func = weight_init

    def optim_create_func( model: Net, lr: float ):
        def get_params( sub_module: Module ,mode="weight"):
            for module in sub_module.modules():
                if isinstance(module, (nn.Conv2d,nn.ConvTranspose2d)):
                    if mode == "weight":
                        yield module.weight
                    elif mode == "bias":
                        if module.bias is not None:
                            yield module.bias
        optimizer = Adam(params=[{'params':get_params(model)},{'params':get_params(model,mode="bias"),'weight_decay':0}], lr=lr,weight_decay=config.WEIGHT_DECAY)
        return optimizer

    def adjust_lr( optimizer:Adam, lr, itr, max_itr ):
        if itr == int(max_itr * config.TRAIN_DOWN_ITER):
            optimizer.state.clear()
            optimizer.param_groups[0]['lr'] = lr * 0.1

    optim_obj = Optimizer(optim_create_func, len(dataset) // batch_size * epoch,
                          step_time_interval=config.STEP_INTERVAL,lr_schuduer=adjust_lr)

    class MyVisualHelper(VisualHelper):

        def __init__( self, catch_time_interval: int ) -> None:
            super(MyVisualHelper, self).__init__(catch_time_interval)
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(pathjoin(config.MODEL_SAVE_ROOT_DIR,'running'))

        def call(self, epoch: int, avg_loss:float, imgs_dict: ImgsType) -> None:
            labels = dataset.real_label(imgs_dict['labels'])
            imgs = dataset.real_image(imgs_dict['imgs'])
            preds = dataset.real_label(torch.sigmoid(imgs_dict['preds']))
            if preds.shape[-2:] != labels.shape[-2:]:
                preds = F.interpolate(preds.float(),size=labels.shape[-2:],mode='bilinear',align_corners=True).type_as(preds)
            show_image = torch.cat([imgs,labels,preds],dim=-1)
            show_image = make_grid(show_image, nrow= 2,pad_value=200)
            self.writer.add_scalar('Loss/train', avg_loss, self._catch_timer)
            self.writer.add_image('Images/pred', show_image, self._catch_timer, dataformats='CHW')

        def close(self) -> None:
            self.writer.close()

    # every epoch see the pic
    model_saver = ModelSaver(len(dataset) // config.BATCH_SIZE * 5, save_dir_path=config.MODEL_SAVE_PATH)

    trainer = Trainer(
        trainloader,
        epoch,
        optim_obj,
        lr,
        device=device,
        loss_function=loss_func,
        visual_helper=MyVisualHelper(len(dataset) // config.BATCH_SIZE // 2) if not config.DISABLE_VISUAL else None,
        model_saver=model_saver,
        pretrained_model_path = config.PRETRAINED_MODEL_PATH,
        weight_init_func = weight_init_func,
        drop_rate=config.DROP_RATE,
    )
    trainer.train()


def test(config: Configuration) -> None:
    if config.DISABLE_TEST:
        return

    dataset = SaliencyDataSet(config.DATASET_TEST_ROOT_DIR,config.DATASET_TEST_LIST_PATH,crop_size=(None,None))
    dataloader = DataLoader(dataset,batch_size=1,shuffle=False,num_workers=1)

    if config.USE_GPU is not None:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    tester = Tester(dataloader,config.TEST_IMG_SAVE_PATH,config.TEST_MODEL_PTH_PATH, \
                    device=device,label_trans_func=lambda x:dataset.real_label(torch.sigmoid(x)).cpu().numpy().transpose((0,2,3,1) )
                    )
    tester.test()

def evaluate(config: Configuration) -> None:
    if config.DISABLE_EVAL:
        return
    if config.USE_GPU is not None:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    sal_measure = get_measure(config.EVAL_MEASURE_LIST, config.EVALUATOR_DIR,config.DATASET_TEST_GT_DIR, device)
    assert 'MAE' in sal_measure.keys()

    measure = OrderedDict()
    measure['setting'] = config.MODEL_SAVE_DIR_NAME
    measure['dataset'] = config.DATASET_NAME
    measure.update(sal_measure)
    for key,item in measure.items():
        if not isinstance(item, Sequence):
            measure[key] = [item]
        else:
            measure[key] = [str(item)]
    table_content = pandas2markdown(pd.DataFrame(measure))
    record_file_dir = os.path.dirname(config.EVALUATOR_SUMMARY_FILE_PATH)
    mkdirs(record_file_dir)

    # export cvv report
    csv_filename = os.path.splitext(os.path.basename(config.EVALUATOR_SUMMARY_FILE_PATH))[0] + '.csv'
    csv_filepath = pathjoin(
        record_file_dir,
        csv_filename
    )

    data_dict = OrderedDict(**{
        'setting':str(config.MODEL_SAVE_DIR_NAME),
        'dataset':str(config.DATASET_NAME),
        'lr':str(config.LEARNING_RATE),
        'epoch':str(config.EPOCH),
        'step_size':str(config.STEP_INTERVAL),
        'optim':str(config.OPTIM),
        'batch_size':str(config.BATCH_SIZE),
        'crop_size':str(config.CROP_SIZE),
        'weight_decay':str(config.WEIGHT_DECAY),
        'drop_rate':str(config.DROP_RATE),
    })
    data_dict.update(sal_measure)
    record_dataframe = pd.DataFrame(data_dict,index=[0])
    if not os.path.exists(csv_filepath):
        record_dataframe.to_csv(csv_filepath,index=False)
    else:
        pd.concat([pd.read_csv(csv_filepath), record_dataframe],sort=False).to_csv(xlsx_filepath,index=False)

    title = config.MODEL_SAVE_DIR_NAME
    file_content = (f"\n"
                    f"# setting {title}  \n"
                    f"time:{time.strftime('%Y-%m-%d %X')}  \n"
                    f"dataset:{config.DATASET_NAME}  \n"
                    f"test dir:{config.EVALUATOR_DIR}  \n"
                    f"command string:\n"
                    f"```bash\n"
                    f"{config.CMD_STR}\n"
                    f"```\n"
                    f"\n"
                    f"## result\n"
                    f"{table_content}\n")
    with portalocker.Lock(config.EVALUATOR_SUMMARY_FILE_PATH, 'a+', \
                          encoding='utf-8',timeout=600) as f:
        f.write(file_content)


def main():
    config = Configuration()
    if config.USE_GPU is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = config.USE_GPU
    if config.PROC_NAME is not None:
        from setproctitle import setproctitle
        setproctitle(config.PROC_NAME)
    train(config)
    for _ in config.update_eval_list():
        test(config)
        evaluate(config)


if __name__ == '__main__':
    main()


