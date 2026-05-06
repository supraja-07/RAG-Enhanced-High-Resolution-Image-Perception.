import torch
import torch.nn as nn


class QueryWeightPredictor(nn.Module):
   """
   Learns adaptive weights for:
   [Base, Neighbor, Object, OCR, Color, Spatial]
   """

   def __init__(self, input_dim=768, hidden_dim=256, num_weights=6):
       super().__init__()

       self.mlp = nn.Sequential(
           nn.Linear(input_dim, hidden_dim),
           nn.ReLU(),
           nn.Linear(hidden_dim, num_weights)
       )

   def forward(self, query_embedding):
       """
       query_embedding: (1, D)
       returns: normalized weights (1, 6)
       """
       weights = self.mlp(query_embedding)
       weights = torch.softmax(weights, dim=-1)
       return weights