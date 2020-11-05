import torch.nn as nn
import torch.nn.init as init



class Block(nn.Module):
    expansion = 2

    def __init__(self, in_planes, cardinality=32, bottleneck_width=4, stride=1):
        super(Block, self).__init__()
        group_width = cardinality * bottleneck_width
        self.conv1 = nn.Conv2d(in_planes, group_width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(group_width)
        self.conv2 = nn.Conv2d(group_width, group_width, kernel_size=3, stride=stride, padding=1, groups=cardinality,
                             bias=False)
        self.bn2 = nn.BatchNorm2d(group_width)
        self.conv3 = nn.Conv2d(group_width, self.expansion * group_width, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * group_width)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * group_width:
            self.shortcut = nn.Sequential(
              nn.Conv2d(in_planes, self.expansion * group_width, kernel_size=1, stride=stride, bias=False),
              nn.BatchNorm2d(self.expansion * group_width)
            )
        self.drop = nn.Dropout()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
      out = self.relu(self.bn1(self.conv1(x)))
      out = self.relu(self.bn2(self.conv2(out)))
      out = self.bn3(self.conv3(out))
      out += self.shortcut(x)
      out = self.relu(out)
      out = self.drop(out)
      return out


class ResNeXt(nn.Module):
  def __init__(self, num_blocks, cardinality, bottleneck_width, num_classes=29, num_channels=3):
    super(ResNeXt, self).__init__()
    self.cardinality = cardinality
    self.bottleneck_width = bottleneck_width
    self.in_planes = 32

    self.conv1 = nn.Conv2d(num_channels, 32, kernel_size=7, bias=False)
    self.bn1 = nn.BatchNorm2d(32)
    self.layer1 = self._make_layer(num_blocks[0], 1)
    self.layer2 = self._make_layer(num_blocks[1], 2)
    self.layer3 = self._make_layer(num_blocks[2], 2)
    self.drop = nn.Dropout(0.3)
    self.linear1 = nn.Linear(cardinality * bottleneck_width * 8, 128)
    self.linear2 = nn.Linear(128, num_classes)
    self.maxpool = nn.MaxPool2d(kernel_size=8)
    self.avgpool = nn.AvgPool2d(kernel_size=8)
    self.relu = nn.ReLU(inplace=True)
    self.drop2 = nn.Dropout(0.4)

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        # init.normal_(m.weight, mean=0.0, std=0.01)
        init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
        if m.bias is not None:
          # init.constant_(m.bias, 0)
          init.ones_(m.bias)

  def _make_layer(self, num_blocks, stride):
    strides = [stride] + [1] * (num_blocks - 1)
    layers = []
    for stride in strides:
      layers.append(Block(self.in_planes, self.cardinality, self.bottleneck_width, stride))
      self.in_planes = Block.expansion * self.cardinality * self.bottleneck_width
    self.bottleneck_width *= 2
    return nn.Sequential(*layers)

  def forward(self, x):
    out = self.relu(self.bn1(self.conv1(x)))
    # out = self.drop(out)
    out = self.layer1(out)
    out = self.layer2(out)
    # out = self.drop(out)
    out = self.layer3(out)
    out = self.avgpool(out)
    out = out.view(out.size(0), -1)
    out = self.drop(out)

    out = self.linear1(out)
    #out = self.drop2(out)
    out = self.linear2(out)
    return out

def custom_resnext(num_classes=10):
  net_args_resnext = {
    "num_blocks": [2, 3, 4],
    "cardinality": 8,
    "bottleneck_width": 4,
    "num_classes": num_classes,
    "num_channels": 3
  }
  model = ResNeXt(**net_args_resnext)
  temp = sum(p.numel() for p in model.parameters() if p.requires_grad)
  print(f'The model architecture:\n\n', model)
  print(f'\nThe model has {temp:,} trainable parameters')
  return model

