import torch
from torch import nn
from torchvision import models

class config:
  num_classes = 120
  labels_path = '../data/labels.csv'
  sample_sub_path = '../data/sample_submission.csv'
  weights_path = '../saved_weights/resnext101.pth'
  size = 256

def get_model(num_classes=120):
  model = models.resnext101_32x8d(pretrained=True)
  num_ftrs = model.fc.in_features
  model.fc = nn.Linear(num_ftrs, num_classes)
  try:
    model.load_state_dict(torch.load(config.weights_path))
  except Exception as e:
    print(f'Exception occured when trying to load weigths:\n{e}')
    return None

  return model
