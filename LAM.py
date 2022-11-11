"""
Interpreting Super-Resolution Networks with Local Attribution Maps
Jinjin Gu, Chao Dong
Project Page: https://x-lowlevel-vision.github.io/lam.html
This is an online Demo. Please follow the code and comments, step by step

First, click file and then COPY you own notebook file to make sure your changes are recorded. Please turn on the colab GPU switch.
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys
import torch

sys.path.append('LAM_Demo-main')
from ModelZoo.utils import Tensor2PIL, PIL2Tensor
from ModelZoo import load_model
from SaliencyModel.utils import vis_saliency, vis_saliency_kde, grad_abs_norm, prepare_images, make_pil_grid
from SaliencyModel.utils import cv2_to_pil, pil_to_cv2, gini
from SaliencyModel.attributes import attr_grad
from SaliencyModel.BackProp import attribution_objective, Path_gradient
from SaliencyModel.BackProp import saliency_map_PG as saliency_map
from SaliencyModel.BackProp import GaussianBlurPath

# model = load_model('CARN@Base')
# model = load_model('EDSR@Base')
# model = load_model('RCAN@Base')
# model = load_model('RNAN@Base')
# model = load_model('RRDBNet@Base')
# model = load_model('SAN@Base')
# model = load_model('HAT@Base')
model = load_model('DBPN@Base')
# model = load_model('SwinIR@Base')

window_size = 16  # Define windoes_size of D
img_lr, img_hr = prepare_images('./test_images/7.png')  # Change this image name
tensor_lr = PIL2Tensor(img_lr)[:3]
tensor_hr = PIL2Tensor(img_hr)[:3]
cv2_lr = np.moveaxis(tensor_lr.numpy(), 0, 2)
cv2_hr = np.moveaxis(tensor_hr.numpy(), 0, 2)

plt.imshow(cv2_hr)

w, h = 110, 150

draw_img = pil_to_cv2(img_hr)
cv2.rectangle(draw_img, (w, h), (w + window_size, h + window_size), (0, 0, 255), 2)
position_pil = cv2_to_pil(draw_img)

sigma = 1.2
fold = 50
l = 9
alpha = 0.5
attr_objective = attribution_objective(attr_grad, h, w, window=window_size)
gaus_blur_path_func = GaussianBlurPath(sigma, fold, l)

interpolated_grad_numpy, result_numpy, interpolated_numpy = Path_gradient(tensor_lr.numpy(), model, attr_objective,
                                                                          gaus_blur_path_func, cuda=True)
grad_numpy, result = saliency_map(interpolated_grad_numpy, result_numpy)
abs_normed_grad_numpy = grad_abs_norm(grad_numpy)
saliency_image_abs = vis_saliency(abs_normed_grad_numpy, zoomin=4)
saliency_image_kde = vis_saliency_kde(abs_normed_grad_numpy)
blend_abs_and_input = cv2_to_pil(
    pil_to_cv2(saliency_image_abs) * (1.0 - alpha) + pil_to_cv2(img_lr.resize(img_hr.size)) * alpha)
blend_kde_and_input = cv2_to_pil(
    pil_to_cv2(saliency_image_kde) * (1.0 - alpha) + pil_to_cv2(img_lr.resize(img_hr.size)) * alpha)
pil = make_pil_grid(
    [position_pil,
     saliency_image_abs,
     blend_abs_and_input,
     blend_kde_and_input,
     Tensor2PIL(torch.clamp(torch.Tensor(result), min=0., max=1.))]
)

pil.show()

gini_index = gini(abs_normed_grad_numpy)
diffusion_index = (1 - gini_index) * 100
print(f"The DI of this case is {diffusion_index}")
