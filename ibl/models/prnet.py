import math
import torch.nn as nn
from torchvision.models import convnext_tiny, efficientnet_b0, regnet_y_800mf
# from ibl.models.vgg import vgg16
import torchvision.models.vgg as tv_vgg


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Conv2d) and classname != "SplAtConv2d":
        nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        init_range = 1.0 / math.sqrt(m.out_features)
        nn.init.uniform_(m.weight, -init_range, init_range)
        nn.init.zeros_(m.bias)


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


def FixBatchNorm(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()
        for p in m.parameters():
            p.requires_grad = False


class Vgg(nn.Module):
    def __init__(self, bb_name, last_dim=128) -> None:
        super().__init__()
        if bb_name == "vgg16":
            ins = tv_vgg.vgg16(pretrained=True)
            conv_layers = list(ins.features.children())[:-2]
            self.last_dim = last_dim
            self.fix_layer_num = 24
            conv_layers[-1] = nn.Conv2d(512, last_dim, kernel_size=3, padding=1)
            weights_init_kaiming(conv_layers[-1])
        elif bb_name == "vgg16_bn":
            ins = tv_vgg.vgg16_bn(pretrained=True)
            conv_layers = list(ins.features.children())[:-2]
            self.last_dim = last_dim
            self.fix_layer_num = 34
            conv_layers[-1] = nn.Conv2d(512, last_dim, kernel_size=3, padding=1)
            weights_init_kaiming(conv_layers[-1])
        self.features = nn.Sequential(*conv_layers)
        self.fix_parameter(self.fix_layer_num)

    def fix_parameter(self, layer_num):
        layers = list(self.features.children())
        for l in layers[:layer_num]:
            for p in l.parameters():
                p.requires_grad = False
                
    def forward(self, x):
        out = self.features(x)
        return out


class PRNet(nn.Module):
    def __init__(self, bb_name, conv_dim=128) -> None:
        super().__init__()
        print("############ use bb_name: {}".format(bb_name))
        self.bb_name = bb_name
        if bb_name == "efficientnet_b0":
            # self.base_model = efficientnet_b0(norm_layer=nn.LayerNorm).features[:6]
            self.base_model = efficientnet_b0().features[:6]
            self.feature_dim = 112
        elif bb_name == "convnext_tiny":
            self.base_model = convnext_tiny().features[:6]
            self.feature_dim =384
            
            for l in self.base_model[:4]:
                for p in l.parameters():
                    p.requires_grad = False
        elif bb_name == "regnet_y_800mf":
            self.base_model = RegnetDM(name=bb_name)
            self.feature_dim =320
        elif bb_name == "vgg16" or bb_name == "vgg16_bn":
            self.base_model = Vgg(bb_name=bb_name, last_dim=conv_dim)
            self.feature_dim = self.base_model.last_dim
        # self.base_model.apply(FixBatchNorm)
    
    def _init_params(self):
        # optional load pretrained weights from matconvnet
        if self.bb_name == "convnext_tiny":
            # self.base_model = convnext_tiny().features[:6]
            # self.feature_dim =384
            # for l in self.base_model[:4]:
            #     for p in l.parameters():
            #         p.requires_grad = False
            pass
            
    def forward(self, x):
        # if self.bb_name == "vgg16":
        #     _, x = self.base_model(x)
        # else:
        x = self.base_model(x)
        return x
    
if __name__ == "__main__":
    import torch
    device = torch.device("cuda:5")
    in_size=[1, 3, 480, 640]
    data = torch.randn(*in_size).to(device)
    net = Vgg(bb_name="vgg16_bn")
    net.to(device)
    out = net(data)
    print(out.shape)
    
    