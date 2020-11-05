import torch.nn as nn
import torch
import torchvision.models as models

class EnsembleThree(nn.Module):
  def __init__(self, num_classes=10):
    super(EnsembleThree, self).__init__()
    self.model1 = models.resnet101(pretrained=True)
    self.model2 = models.resnext50_32x4d(pretrained=True)
    self.model3 = models.densenet201(pretrained=True)
    self.model4 = models.shufflenet_v2_x2_0(pretrained=True)
    self.models = [self.model1, self.model2, self.model3]
    for model in self.models:
      for param in model.parameters():
        param.requires_grad = False
    self.feats1 = self.model1.fc.in_features

    self.feats2 = self.model2.fc.in_features

    self.feats3 = self.model3.classifier.in_features
    self.model1.fc = nn.Identity()
    self.model2.fc = nn.Identity()
    self.model3.classifier = nn.Identity()


    self.relu = nn.ReLU(inplace=True)
    self.classifier = nn.Linear(self.feats1 + self.feats2 + self.feats3, num_classes)

  def forward(self, x):
    x1 = self.model1(x.clone())
    x1 = x1.view(x1.size(0), -1)

    x2 = self.model2(x.clone())
    x2 = x2.view(x2.size(0), -1)

    x3 = self.model3(x)
    x3 = x3.view(x3.size(0), -1)

    x = torch.cat((x1, x2, x3), dim=1)
    x = self.classifier(self.relu(x))
    return x


def get_ensemble(num_classes):
  model = EnsembleThree(num_classes)
  temp = sum(p.numel() for p in model.parameters() if p.requires_grad)
  print(f'The model architecture:\n\n', model)
  print(f'\nThe model has {temp:,} trainable parameters')
  return model


