import torch
import torch.nn as nn
import torch.nn.functional as F

class NormalizedModel(nn.Module):
    def __init__(self, base_model, mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)):
        super().__init__()
        self.base_model = base_model
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x, task_id=0):
        return self.base_model((x - self.mean) / self.std, task_id=task_id)

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, 3, stride=1, padding=1, bias=False)
        self.droprate = dropRate
        self.shortcut = (
            nn.Conv2d(in_planes, out_planes, 1, stride=stride, bias=False)
            if stride != 1 or in_planes != out_planes else None
        )

    def forward(self, x):
        out = self.relu1(self.bn1(x))
        residual = self.shortcut(out) if self.shortcut is not None else x
        out = self.conv1(out)
        out = self.relu2(self.bn2(out))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        return residual + self.conv2(out)

class WideResNet(nn.Module):
    def __init__(self, depth=28, num_classes=10, widen_factor=10, dropRate=0.0, num_tasks=2):
        super().__init__()
        n = (depth - 4) // 6
        channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]

        self.conv1 = nn.Conv2d(3, channels[0], 3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(channels[0], channels[1], n, 1, dropRate)
        self.layer2 = self._make_layer(channels[1], channels[2], n, 2, dropRate)
        self.layer3 = self._make_layer(channels[2], channels[3], n, 2, dropRate)
        self.bn = nn.BatchNorm2d(channels[3])
        self.relu = nn.ReLU(inplace=True)

        # Head 0: Clean / Standard Task | Head 1: GPM Adversarial Task
        self.heads = nn.ModuleList([nn.Linear(channels[3], num_classes) for _ in range(num_tasks)])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, in_planes, out_planes, num_blocks, stride, dropRate):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(in_planes, out_planes, s, dropRate))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x, task_id=0):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.relu(self.bn(out))
        out = F.adaptive_avg_pool2d(out, (1, 1)).flatten(1)
        return self.heads[task_id](out)