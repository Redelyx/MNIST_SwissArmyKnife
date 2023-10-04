import torch.nn as nn

class UpdatedLeNet(nn.Module):
    def __init__(self, n_feature, output_size):
        super(UpdatedLeNet, self).__init__()
        self.n_feature = n_feature
        self.output_size = output_size

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=n_feature, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=n_feature, out_channels=16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.fc1 = nn.Linear(16*4*4, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, output_size)  # Output layer for 3-way classification

    def forward(self, x):
        x = self.conv_block1(x) 
        x = self.conv_block2(x) 
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        x = self.relu4(x)
        logits = self.fc3(x)
        return logits
