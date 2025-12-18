import torch.nn as nn
import torch.nn.functional as F

class LeNetClassifier(nn.Module):
    def __init__(self, num_classes, in_channels=1, img_size=28):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, 6, 5, padding="same")
        self.avgpool1 = nn.AvgPool2d(2)

        self.conv2 = nn.Conv2d(6, 16, 5)
        self.avgpool2 = nn.AvgPool2d(2)

        if img_size == 28:
            fc_input = 16 * 5 * 5
        else:
            fc_input = 16 * 35 * 35

        self.fc1 = nn.Linear(fc_input, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.avgpool1(x)
        x = F.relu(self.conv2(x))
        x = self.avgpool2(x)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
