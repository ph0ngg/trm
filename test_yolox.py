from models.yolox import YOLOX

import cv2
import numpy as np
import torch

model = YOLOX()
model.load_state_dict(torch.load('/mnt/data_ubuntu/phongnn/yolox_l.pth')['model'])
path_to_img1 = '/mnt/data_ubuntu/phongnn/MOT17/images/train/MOT17-02-SDP/img1/000104.jpg'
path_to_img2 = '/mnt/data_ubuntu/phongnn/MOT17/images/train/MOT17-04-SDP/img1/000230.jpg'


img1 = cv2.imread(path_to_img1)
img1 = cv2.resize(img1, (1088, 608))
img1 = np.transpose(img1, (2, 0, 1))
img1 = np.expand_dims(img1, 0)
img1 = torch.Tensor(img1)

img2 = cv2.imread(path_to_img2)
img2 = cv2.resize(img2, (1088, 608))
img2 = np.transpose(img2, (2, 0, 1))
img2 = np.expand_dims(img2, 0)
img2 = torch.Tensor(img2)

imgs = torch.cat((img1, img2), 0)
print(imgs.shape)
model.cuda().eval()

_, yolo_outputs, _ = model(imgs.cuda())
print(yolo_outputs)
