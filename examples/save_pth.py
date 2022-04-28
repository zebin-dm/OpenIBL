# base_model = models.create(args.arch, train_layers=args.layers,
#                            matconvnet='logs/vd16_offtheshelf_conv5_3_max.pth')
import torch
import ibl.models as models
ARCH = 'prnet'
bb_name = "vgg16"
conv_dim = 128
pth_file = "/data/zebin/OpenIBL/logs/netVLAD/pitts250k-prnet/vgg16-sare_ind-lr0.001-tuple8-cd128-rd4096-SFRS/model_best.pth.tar"

base_model = models.create(ARCH, bb_name=bb_name, conv_dim=conv_dim)
pool_layer = models.create('netvlad', dim=conv_dim, parafile=None)
model = models.create('embedregionnet', base_model, pool_layer,
                      tuple_size=1, reduce=False, reduce_dim=-1)
state_dict = torch.load(pth_file)
model.load_state_dict(state_dict['state_dict'])
model.base_model.
