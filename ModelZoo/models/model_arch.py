import torch

# model_path1 = 'rcan_x2_train3_epoch-best.pth'
model_path1 = './swin_tiny_patch4_window7_224.pth'
model1 = torch.load(model_path1)
model2 = {}
for key in model1:
    print(key)

# print(model2)
# print(model1)
# print(model1['model']['name'])
# print(model1['model']['args'])  # {'scale': 2}
# print(model1['model']['sd'])

# dict = {
#     'model': {
#         'name': 'hat',
#         'args': {'scale': 2},
#         'sd': model1['params_ema']
#     }
# }

# torch.save(model1['params_ema'], 'HAT_SRx4.pth')
# torch.save(model2, "DBPN_x4.pth")
# print(model1['params_ema'])
# torch.save(model1['model'], 'swin_tiny_patch4_window7_224.pth')
