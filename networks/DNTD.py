from typing import Tuple
from config import Configuration
from efficientnet_pytorch import EfficientNet
from helper.utils import file_exist
from typing import Union,List
from base_model.ResNet import resnet50

import torch
import torch.nn as nn
from torch.nn import functional as F
import timm


class PPM(nn.Module):
    """
    Spatial Pyramid Pooling
    """

    def __init__( self, dim_in, dim_out, trans_outs):
        super().__init__()
        self.ppms = nn.ModuleList()
        self.ppms.append(nn.Sequential(nn.Conv2d(dim_in, dim_out, 1, 1, bias=False),
                                       nn.ReLU(inplace=True)))
        for ii in [1, 3, 5, 7]:
            self.ppms.append(nn.Sequential(nn.AdaptiveAvgPool2d(ii), nn.Conv2d(dim_in, dim_out, 1, 1, bias=False),
                              nn.ReLU(inplace=True)))
        self.conv_cats = nn.ModuleList()
        for channel in trans_outs:
            self.conv_cats.append(nn.Sequential(
                nn.Conv2d(dim_out * 5, channel, 1, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
            ))

    def forward(self, x):
        [b,c,row,col] = x.size()
        ppm_outs = [self.ppms[0](x),]
        for module in self.ppms[1:]:
            ppm_outs.append(F.interpolate(module(x), (row,col), mode='bilinear', align_corners=True))
        feature_cat = torch.cat(ppm_outs, dim=1)
        result = list()
        for module in self.conv_cats:
            result.append(module(feature_cat))
        return result


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('relu', nn.ReLU(inplace=False))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=True))

    def eval(self):
        self._module['relu'] = nn.ReLU(inplace=True)
        return super().eval()


class PlainConv(nn.Module):
    def __init__( self, num_layers, num_input_features, bn_size, growth_rate, num_out_features, drop_rate=0 ):
        super(PlainConv, self).__init__()
        self.conv1 = nn.Conv2d(num_input_features, num_out_features, 3, 1, 1)

    def forward( self, x ):
        x = F.relu(x, inplace=not self.training)
        x = self.conv1(x)
        return x


class ConcatTransition(nn.Module):

    def __init__(self,num_input_features,num_out_features):
        super(ConcatTransition,self).__init__()
        self.transition = _Transition(num_input_features, num_out_features)

    def forward(self, input):
        if isinstance(input, torch.Tensor):
            input_features = [input]
        else:
            input_features = input
        input_features = torch.cat(input_features,dim=1)
        return self.transition(input_features)



class DNTD(nn.Module):

    def __init__(self,encoder_out_channels,encoder_new_out_channels,drop_rate = 0):
        super(DNTD,self).__init__()

        self.group_transitions = nn.ModuleList([_Transition(encoder_out_channel,out_encoder_channel) for encoder_out_channel,out_encoder_channel in zip(encoder_out_channels,encoder_new_out_channels)])
        ppm_in = encoder_out_channels[0]
        encoder_out_channels = encoder_new_out_channels

        group1_channels = list(encoder_new_out_channels[0:])
        group2_channels = group1_channels.copy()
        ppm_channels = group1_channels[1:]

        # self.aspp actually is PPM
        self.aspp = PPM(ppm_in, encoder_out_channels[0] // 4, ppm_channels)

        self.denseblocks = nn.ModuleList([PlainConv(2, group2_channel, 2, \
                    group2_channel // 4, out_channel, drop_rate=drop_rate)
                    for i,(group2_channel,out_channel) in enumerate(zip(group2_channels[0:-1],group2_channels[1:]))])

        self.group1_channels = group1_channels
        self.group2_channels = group2_channels
        group1_transitions = nn.ModuleDict()
        self.count = len(encoder_out_channels)
        for i in range(1, self.count):
            transitions = nn.ModuleList()
            for in_channel, out_channel in zip(group1_channels[i - 1:-1], group1_channels[i:]):
                transitions.append(_Transition(in_channel, out_channel))
            group1_transitions.add_module('group1_transitions_%d' % i,transitions)

            # next four lines are not meaningful
            dw_convs = nn.ModuleList()
            for in_channel, out_channel in zip(group1_channels[i - 1:-1], group1_channels[i:]):
                dw_convs.append(nn.Sequential(nn.ReLU(inplace=True),nn.Conv2d(out_channel,out_channel,3,1,1,groups=out_channel)))
            group1_transitions.add_module('dw_convs_%d' % i, dw_convs)

        self.group1_transitions = group1_transitions
        group1_cancat_transitions = nn.ModuleList()
        for i, group1_channel in enumerate(group1_channels[1:]):
            cancat_transition = ConcatTransition( (2 + i) * group1_channel, group1_channel)
            group1_cancat_transitions.append(cancat_transition)
        self.group1_cancat_transitions = group1_cancat_transitions

        group2_cancat_transitions = nn.ModuleList()
        for i, (ppm_channel,group1_channel, group2_channel, out_channel) in enumerate(
                zip(ppm_channels[0:-1],group1_channels[1:-1], group2_channels[1:-1], group2_channels[1:-1])):
            cancat_transition = ConcatTransition(ppm_channel + group1_channel + 2 * group2_channel, out_channel)
            group2_cancat_transitions.append(cancat_transition)

        group2_cancat_transitions.append(
                ConcatTransition(ppm_channels[-1] + group1_channels[-1] + 2 * group2_channels[-1], 4))
        self.group2_cancat_transitions = group2_cancat_transitions
        self.score = nn.Conv2d(4,1,1)


    def upsample(self,x:torch.Tensor,size:Tuple[int,int]) -> Union[torch.Tensor,List[torch.Tensor]]:
        if isinstance(x,(list,tuple)):
            x_s = list()
            for _x in x:
                if _x.shape[-2:] == size:
                    x_s.append(_x)
                else:
                    x_s.append(F.interpolate(_x, size=size, mode='bilinear', align_corners=True))
            return x_s
        else:
            if x.shape[-2:] == size:
                return x
            return F.interpolate(x,size=size,mode='bilinear',align_corners=True)

    def forward(self, inputs, x_size):
        ppm_outs = self.aspp(F.relu(inputs[0], inplace=not self.training))
        inputs = [module(x) for module,x in zip(self.group_transitions,inputs)]

        group1_inputs = inputs
        group2_inputs = inputs
        top_group1_input = group1_inputs[0]
        top_group2_input = group2_inputs[0]

        # progressive compres-sion  shortcut  paths
        inputs = [top_group1_input]
        group1_out_list = []
        dst_size_list = list()
        for i in range(1,self.count):
            dst_size = group1_inputs[i].shape[-2:]
            dst_size_list.append(dst_size)
            out_list = []
            col = i
            for j,input in enumerate(inputs):
                out = self.group1_transitions['group1_transitions_%d' % (j+1)][col-1](input)
                out = self.upsample(out,dst_size)
                out_list.append(out)
                col -= 1

            out_list.append(group1_inputs[i])
            group1_out_list.append(out_list)
            inputs = list(out_list)


        group1_concat_out_list = []
        for dst_size,input,module in zip(dst_size_list,group1_out_list,self.group1_cancat_transitions):
            new_input = input
            group1_concat_out_list.append(module(new_input))

        # fusion process
        input = top_group2_input
        for i in range(1,self.count):
            dst_size = group2_inputs[i].shape[-2:]
            db_out = self.denseblocks[i-1](input)
            db_out = self.upsample(db_out,dst_size)
            inputs = [self.upsample(ppm_outs[i-1],dst_size),group1_concat_out_list[i-1],db_out,group2_inputs[i]]
            input = self.group2_cancat_transitions[i-1](inputs)

        db_out = F.relu(input, inplace=True)
        score_map = self.score(db_out)
        return self.upsample(score_map,size=x_size)

class Net(nn.Module):
    def __init__(self,drop_rate=0):
        super(Net,self).__init__()

        self.base_modelname = Configuration.instance().MODEL_NAME
        base_modelname = self.base_modelname
        if base_modelname == "resnet50":
            self.encoder = resnet50(pretrained=False)
            encoder_out_channels = (2048, 1024, 512, 256, 64)
            encoder_new_out_channels = (int(2048 / Configuration.instance().RESNET_SCALE),
                                        int(1024 / Configuration.instance().RESNET_SCALE),
                                        int(512 / Configuration.instance().RESNET_SCALE),
                                        int(256 / Configuration.instance().RESNET_SCALE),
                                        int(64 / Configuration.instance().RESNET_SCALE))
            self.decoder: DNTD = DNTD(encoder_out_channels, encoder_new_out_channels,drop_rate=drop_rate)
        elif base_modelname == "efficientnet-b3":
            self.encoder:EfficientNet = EfficientNet.from_name('efficientnet-b3',override_params={'num_classes': None})
            encoder_out_channels = (1536,96,48,32,40)
            encoder_new_out_channels = (1536 // 8,96 // 2,48 // 2,32 // 2,32 // 4)
            self.decoder:DNTD = DNTD(encoder_out_channels,encoder_new_out_channels,drop_rate=drop_rate)
        elif base_modelname == "efficientnet-b0":
            self.encoder:EfficientNet = EfficientNet.from_name('efficientnet-b0',override_params={'num_classes': None})
            encoder_out_channels = (1280,80,40,24,32)
            encoder_new_out_channels = (1280 // 8,80 // 2,40 // 2,24 // 2,32 // 2)
            self.decoder:DNTD = DNTD(encoder_out_channels,encoder_new_out_channels,drop_rate=drop_rate)
        else:
            raise Exception("No model selected")

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def load_encoder_weight(self,pth_path):
        device = self.decoder.score.weight.device
        self.to('cpu')

        ret = "No pretrained file"
        if file_exist(pth_path):
            state_dict = torch.load(open(pth_path, 'rb'), map_location='cpu')
            base_modelname = self.base_modelname
            if base_modelname.startswith('efficientnet'):
                state_dict.pop('_fc.weight')
                state_dict.pop('_fc.bias')
            elif base_modelname.startswith('resnet'):
                state_dict.pop('fc.weight')
                state_dict.pop('fc.bias')
            ret = self.encoder.load_state_dict(state_dict, strict=False)
            del state_dict
        self.to(device)
        return ret

    def forward(self, x):
        outs = self.encoder(x)
        out = self.decoder(outs, x.shape[-2:])
        return out

if __name__ == '__main__':
    import sys
    sys.argv.append('-d')
    sys.argv.append('SOD')
    sys.argv.append('-save')
    sys.argv.append('test')
    sys.argv.append('-model_name')
    sys.argv.append('efficientnet-b0')
    import cv2
    x = torch.rand(1,3,500,500)
    '''
    
    import numpy as np
    img = cv2.imread('../123.jpg')
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB).astype(np.float32)
    img_mean: np.ndarray = np.array([0.485, 0.456, 0.406]).reshape(1,1,3)
    img_std: np.ndarray = np.array([0.229, 0.224, 0.225]).reshape(1,1,3)
    img /= 255.0
    img -= img_mean
    img /= img_std
    img = np.transpose(img,[2,0,1])
    x = torch.from_numpy(img).unsqueeze(0)
    '''

    model:Net = Net()
    ret = model.load_state_dict(torch.load('../pretrained_model/dntd-efficient-b0.pth',map_location='cpu')['model'])
    print(ret)
    model.eval()
    out = model(x)
    out = torch.sigmoid(out[0][0])*255
    out = out.to(torch.uint8)
    cv2.imshow("test",out.numpy())
    cv2.waitKey(0)
    print(out.shape)


