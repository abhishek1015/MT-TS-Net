import torchvision.models as models
import torch

class customResnetModel(torch.nn.Module):
    def __init__(self):
        super(customResnetModel,self).__init__()
        self.model = models.resnet18(pretrained=True)       
        self.model.fc = torch.nn.Linear(in_features=512, out_features=512, bias=True)

    def _run_step(self,x):
        out1 = self.model(x)
        return out1
