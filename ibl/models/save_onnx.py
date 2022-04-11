
import torch
from ibl.models.netvlad import EmbedRegionNet
from ibl.models.prnet import PRNet
from ibl.models.netvlad import NetVLAD


def save_model():
    bb_name = "vgg16"
    conv_dim = 128
    device = torch.device("cuda:0")
    ck_file = "/data/zebin/OpenIBL/logs/netVLAD/pitts250k-prnet/vgg16-sare_ind-lr0.001-tuple8-SFRS/model_best.pth.tar"
    save_file = "/data/zebin/OpenIBL/logs/netVLAD/pitts250k-prnet/vgg16-sare_ind-lr0.001-tuple8-SFRS/reginnet.pth"
    base_model = PRNet(bb_name=bb_name, conv_dim=conv_dim)
    pool_layer  = NetVLAD(num_clusters=64, dim=conv_dim)
    net = EmbedRegionNet(base_model=base_model, net_vlad=pool_layer, reduce=True)
    state_dict = torch.load(ck_file, torch.device("cpu"))
    
    
    pnet = torch.nn.DataParallel(net)
    pnet.load_state_dict(state_dict=state_dict["state_dict"])
    pnet.cpu()
    torch.save(pnet.module.state_dict(), save_file)
    


def save_onnx():
    bb_name = "vgg16"
    conv_dim = 128
    device = torch.device("cuda:0")
    ck_file = "/data/zebin/OpenIBL/logs/netVLAD/pitts250k-prnet/vgg16-sare_ind-lr0.001-tuple8-SFRS/reginnet.pth"
    onnx_file = "/data/zebin/OpenIBL/logs/netVLAD/pitts250k-prnet/vgg16-sare_ind-lr0.001-tuple8-SFRS/reginnet.onnx"
    base_model = PRNet(bb_name=bb_name, conv_dim=conv_dim)
    pool_layer  = NetVLAD(num_clusters=64, dim=conv_dim)
    net = EmbedRegionNet(base_model=base_model, net_vlad=pool_layer, reduce=True)
    state_dict = torch.load(ck_file, torch.device("cpu"))
    net.load_state_dict(state_dict=state_dict)
    # net.to(device)
    net.eval()

    in_names = ["data",]
    out_names = ["feature",]
    in_size=[1, 3, 480, 640]
    data = torch.randn(*in_size)
    with torch.no_grad():
        out_data = net(data)
        print(out_data.shape)
        dynamic_axes = {in_names[0]: {0: 'batch'},
                        out_names[0]: {0: 'batch'}}

        torch.onnx.export(net,
                         data,
                         onnx_file,
                         input_names=in_names,
                         output_names=out_names,
                         dynamic_axes=dynamic_axes,
                         export_params=True,
                         opset_version=10,
                         do_constant_folding=True,
                         verbose=True)

if __name__ == "__main__":
    save_onnx()
    # save_model()