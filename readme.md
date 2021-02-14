# Revisiting Top-Down Flows for Salient Object Detection

This repository is the implementation of **Revisiting Top-Down Flows for Salient Object Detection**. 

## Prerequisites

- [Pytorch 1.7.0+](http://pytorch.org/)
- [Python3.6+](http://python.org/)

## Requirements

> Note: Please **clone this project** and  install required **pytorch** first

To install requirements:

```setup
pip install -r requirements.txt
```

### Backbone  pre-trained model download 
Please select one of the links below to download pre-trained models
- [BaiduDisk](https://pan.baidu.com/s/1Iv_J33DiqkrANINc2sig1g): (code: 1234)
- [GoogleDisk](https://drive.google.com/drive/folders/1fHK-ljCCcK86iFoPJ28rGoRcx4qzZrPt?usp=sharing)

After downloading, put it into `pretrained_model` folder

### Dataset download 
Please select one of the links below to download related  Saliency Object Detection Dataset
- [BaiduDisk](https://pan.baidu.com/s/1Iv_J33DiqkrANINc2sig1g): (code: 9ib7)
- [GoogleDisk](https://drive.google.com/file/d/1LEc5s_gJ8AkdcXr6KFtJQ9ffPUE3NtUW/view?usp=sharing)

After downloading, use `7z -x filepath` to extract `7z` file into current project folder

## Train and test

To train the model, run following commmands:

```bash
# train efficient-b3 backbone
python run.py -d DUTS -pretrain ./pretrained_model/efficientnet-b3-5fb5a3c3.pth -save efficient-b3 -optim Adam -epoch 210 -lr 4.5e-5 -gpu 0 -b 1 -crop 0 -step 10 -proc DNTD -weight_decay 0 -is_scale -random_bright --TRAIN_DOWN_ITER 0.8 -model_name efficientnet-b3
# train efficient-b0 backbone
python run.py -d DUTS -pretrain ./pretrained_model/efficientnet-b0-355c32eb.pth -save efficient-b0 -optim Adam -epoch 210 -lr 4.5e-5 -gpu 0 -b 1 -crop 0 -step 10 -proc DNTD -weight_decay 0 -is_scale -random_bright --TRAIN_DOWN_ITER 0.8 -model_name efficientnet-b0  
# train resnet-50 backbone
python run.py -d DUTS -pretrain ./pretrained_model/resnet.pth -save resnet-scale-4 -epoch 210 -lr 1e-5 -gpu 0 -b 1 -crop 0 -step 10 -proc DNTD -weight_decay 0 -is_scale -random_bright --TRAIN_DOWN_ITER 0.8 -model_name resnet50  --RESNET_SCALE 4
```

> Note: 
>
> 1. **The commands will automatically test and evaluate the trained model**.
>
>    If you need not to evaluate, you can use the `-disable_eval` parameter.
>
>    If you need not to test,  you can use the `-disable_test` parameter.
>
> 2. `-gpu` parameter is the GPU number that you use.(e.g. `0` `0,1`...)
>
> 3. Evaluation code is embedded in this project. 
>
>    If you want to evaluate all dataset, you can use the `-eval_d All` parameter.
>
> 4. The evaluation result can be found in  `logs/ExperimentalNotes.md` 


## Evaluation

- To evaluate other trained model on the Saliency Object Detection Dataset, run:

```bash
python run.py -d DUTS -save efficient-b3 -model_name efficientnet-b3 -test_model ./dntd-efficient-b3.pth -disable_train -eval_d ALL 
```

- If you want the obtained results using this project, you need to adapt your folder names to our required structure and put it into `logs` directory. 
Suppose your directory name is "SOD", you can specify `-save SOD `  and `-disable_train -diabale_test`, then run the following command to evaluate

```bash 
python run.py -d DUTS -save <Your folder name in log dir> -disable_train -disable_test -eval_d ALL 
```

### Required evaluation folder structure

```
logs/
|-- <folder name>
|   `-- test
|       |-- SOD
|       |-- DUTS
|       `-- ...
```


## Pre-trained SOD Models and Predicted Saliency map

You can download our pretrained SOD models here:
- [BaiduDisk](https://pan.baidu.com/s/1VP2yiFFWQSeZ9UZ5nMoXTQ): (code: 1234)
- [GoogleDisk](https://drive.google.com/drive/folders/1a9dQjwr6bCAWRmFCSRTuf1d8kgN4viNr?usp=sharing)

We also release our predicted saliency maps:

- [BaiduDisk](https://pan.baidu.com/s/1C1yofH-Y5B99o1KU7uWEdQ): (code: 1234)
- [GoogleDisk](https://drive.google.com/drive/folders/1dFysx3snmocqYN8uzH0iOay9iplanI9V?usp=sharing)


