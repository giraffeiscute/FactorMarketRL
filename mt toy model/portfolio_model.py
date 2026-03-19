import torch.nn as nn

class PortfolioModel(nn.Module):
    def __init__(self, d_model):
        super(PortfolioModel, self).__init__()
        self.layer1 = nn.Linear(d_model, 32)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(32, 1)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, embeddings):
        x = self.relu(self.layer1(embeddings))
        logits = self.layer2(x)
        weights = self.softmax(logits)
        return weights.squeeze()