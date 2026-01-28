import torch
import torchvision
import torch.nn as nn
from discrim_model import DiscriminatorModel
from gen_model import GenerativeModel


"""
GanModel
Currently a placeholder class, not sure if we need this? 
may be useful for training to compartmentalize
"""
class GanModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.disc = DiscriminatorModel()
        self.gen = GenerativeModel()
