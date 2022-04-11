from torch import nn
# from torchvision.models import efficientnet_v2_s
from ibl.models.netvlad import NetVLAD
from ibl.models.vgg import vgg16
from torchvision.models import convnext_tiny, efficientnet_b0, regnet_y_800mf

import torch.nn.functional as F


class VggVlad(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        from ibl.models.netvlad import NetVLAD
        self.base_model = vgg16()
        self.pool = NetVLAD()
    
    def forward(self, x):
        pool_x, x = self.base_model(x)
        print(x.shape)
        vlad_x = self.pool(x)
        # normalize
        vlad_x = F.normalize(vlad_x, p=2, dim=2)  # intra-normalization
        vlad_x = vlad_x.view(x.size(0), -1)  # flatten
        vlad_x = F.normalize(vlad_x, p=2, dim=1)  # L2 normalize
        return vlad_x


class RegnetDM(nn.Module):
    def __init__(self, name) -> None:
        super().__init__()
        if name == 'regnet_y_800mf':
            ins = regnet_y_800mf()
        self.stem = ins.stem
        self.feat = ins.trunk_output[:3]
        print("self.feat layers: {}".format(len(self.feat)))
    
    def forward(self, x):
        out = self.stem(x)
        out = self.feat(out)
        return out


class PRNet(nn.Module):
    def __init__(self, name) -> None:
        super().__init__()
        self.name = name
        if name == "vgg16":
            self.base_model = vgg16()
            self.base_dim = 512
        elif name == "efficientnet_b0":
            self.base_model = efficientnet_b0().features[:6]
            self.base_dim = 112
        elif name == "convnext_tiny":
            self.base_model = convnext_tiny().features[:6]
            self.base_dim =384
        elif name == "regnet_y_800mf":
            self.base_model = RegnetDM(name=name)
            self.base_dim =320
        
        self.pool = NetVLAD(dim=self.base_dim)
        
    
    def forward(self, x, debug=False):
        if self.name == "vgg16":
            _, x = self.base_model(x)
        else:
            x = self.base_model(x)
        if debug:
            print(x.shape)
            assert x.shape[2] == 30
            assert x.shape[3] == 40
        vlad_x = self.pool(x)
        vlad_x = F.normalize(vlad_x, p=2, dim=2)  # intra-normalization
        vlad_x = vlad_x.view(x.size(0), -1)  # flatten
        vlad_x = F.normalize(vlad_x, p=2, dim=1)  # L2 normalize
        return vlad_x


def save_model():
    torch.onnx.export(self.net,
                args=self.dummy_input,
                f=model_save_path,
                input_names=self.in_names,
                output_names=self.out_names,
                dynamic_axes=None,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                verbose=True)


def run_test(name, data, device, save_model=False):
    print("\n eval model: {}".format(name))
    net = PRNet(name=name)
    net.eval()
    net.to(device)
    with torch.no_grad():
        out = net(data, debug=True)
    print(out.shape)
    
    if save_model:
        in_names = ["data", ]
        out_names = ["feat",]
    

def time_test(name, data, device):
    print("\n time test, model: {}".format(name))
    net = PRNet(name=name)
    net.eval()
    net.to(device)
    net(data)
    run_time = 1000
    torch.cuda.synchronize(device)
    start_time = time.time()
    with torch.no_grad():
        for idx in range(run_time):
            net(data)
    time_interval = time.time() - start_time
    torch.cuda.synchronize(device)
    fps = 1000 / time_interval
    print("fps: {}".format(fps))

if __name__ == "__main__":
    all_model = ['vgg16', 'efficientnet_b0', 'convnext_tiny', 'regnet_y_800mf']
    import torch
    import time
    device = torch.device("cuda:5")
    in_size=[1, 3, 480, 640]
    data = torch.randn(*in_size).to(device)
    for model_name in all_model:
        run_test(name=model_name,
                 data=data,
                 device=device)
        
        time_test(name=model_name,
                  data=data,
                  device=device)

    