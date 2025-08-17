import torch.nn as nn

class CaeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cae = nn.Sequential(
            nn.Conv1d(7, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2, padding=1),
            nn.Conv1d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2, padding=1),
            nn.ConvTranspose1d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.cae(x)
        return x