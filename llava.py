import torch
import torch.nn as nn


class RAPTrainer:

   def __init__(self, model):

       self.model = model

       self.ce_loss = nn.CrossEntropyLoss()

   def compute_loss(
       self,
       logits,
       target
   ):

       ce = self.ce_loss(
           logits,
           target
       )

       return ce