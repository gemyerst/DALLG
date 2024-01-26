
from torch import Tensor
from torch import nn


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_hidden: int):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_model)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor):
        return self.fc2(self.relu(self.fc1(x)))
