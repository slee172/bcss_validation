'''Modification from

@article{honzatko2021defect,
  title={Defect segmentation for multi-illumination quality control systems},
  author={Honz{\'a}tko, David and T{\"u}retken, Engin and Bigdeli, Siavash A and Dunbar, L Andrea and Fua, Pascal},
  journal={Machine vision and Applications},
  year={2021},
  volume={32},
  number={118},
  publisher={Springer}
}
'''

import abc


class Net:
    def __init__(self):
        self.model = None
        __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def net(self):
        """Returns the keras model of the network"""
        return

    @abc.abstractmethod
    def get_name(self):
        """Returns the string specifying the network"""
        return

    def summary(self):
        """Returns the keras model summary"""
        if self.model is not None:
            return self.model.summary()
        else:
            return None
