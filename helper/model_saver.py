from torch.nn import Module
import torch
from helper.utils import mkdirs
from helper.utils import pathjoin
import time


class ModelSaver(object):

    def __init__(self,save_interval,save_dir_path=None,save_base_name='model'):
        self.__timer = 0
        self.__save_interval = save_interval
        self.__save_dir_path = save_dir_path
        if self.__save_dir_path is None:
            self.__save_dir_path = pathjoin(
                '../',
                time.strftime("%F %H-%M-%S",time.localtime())
            )
        mkdirs(self.__save_dir_path)
        self.__base_model_name = save_base_name
        self.__interval_timer = 0

    def state_dict(self):
        return {
            '__timer':self.__timer,
            '__save_interval':self.__save_interval,
            '__interval_timer':self.__interval_timer,
        }

    def load_state_dict(self,state_dict):
        self.__timer = state_dict['__timer']
        self.__save_interval = state_dict['__save_interval']
        self.__interval_timer = state_dict['__interval_timer']

    def __call__(self, model_dict, isFinal:bool = False,hooks=False):
        self.__timer += 1
        if self.__timer % self.__save_interval == 0 or isFinal:
            self.__interval_timer += 1
            model_ext_name = 'final' if isFinal else str(self.__interval_timer)
            model_save_path = pathjoin(
                self.__save_dir_path,
                self.__base_model_name+'-'+model_ext_name+'.pth'
            )
            torch.save(model_dict,model_save_path)
            if hooks:
                hooks(model_save_path,self.__interval_timer)

