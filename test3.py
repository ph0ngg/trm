import torch
import cv2

from numpy.linalg import norm

from models.yolox import *

model1 = YOLOX()
model2 = Head2()

model1.load_state_dict(torch.load('/mnt/data_ubuntu/phongnn/yolox_l.pth')['model'])
model2.load_state_dict(torch.load('/mnt/data_ubuntu/phongnn/Towards-Realtime-MOT/weights/run22_08_06_22/latest.pt')['model'])

path_to_img1 = '/mnt/data_ubuntu/phongnn/MOT17/images/train/MOT17-02-DPM/img1/000001.jpg'

path_to_img2 = '/mnt/data_ubuntu/phongnn/MOT17/images/train/MOT17-02-DPM/img1/000002.jpg'

model1.eval()
model2.eval()

img1 = cv2.imread(path_to_img1)
img2 = cv2.imread(path_to_img2)
img1 = cv2.resize(img1, (1088, 608))
img1 = np.transpose(img1, (2, 0, 1))

img1 = np.expand_dims(img1, 0)
img1 = torch.Tensor(img1)

img2 = cv2.resize(img2, (1088, 608))
img2 = np.transpose(img2, (2, 0, 1))

img2 = np.expand_dims(img2, 0)
img2 = torch.Tensor(img2)

xin, yolo_outputs, reid_idx = model1(img1)
filtered_outputs = []
filtered_outputs_reid_idx = []
for batch_idx in range(len(yolo_outputs)):
    batch_data = yolo_outputs[batch_idx]
    batch_reid_idx = reid_idx[batch_idx]
    class_mask = batch_data[:, 6] == 0
    filtered_output = batch_data[class_mask, :]
    filtered_reid_idx = batch_reid_idx[class_mask]
    filtered_outputs.append(filtered_output)
    filtered_outputs_reid_idx.append(filtered_reid_idx)
print(yolo_outputs[0][1])
print(filtered_outputs_reid_idx[0].shape)
_, emb_vt1 = model2(xin, filtered_outputs, filtered_outputs_reid_idx)
print(len(emb_vt1))

xin, yolo_outputs, reid_idx = model1(img2)
filtered_outputs = []
filtered_outputs_reid_idx = []
for batch_idx in range(len(yolo_outputs)):
    batch_data = yolo_outputs[batch_idx]
    batch_reid_idx = reid_idx[batch_idx]
    class_mask = batch_data[:, 6] == 0
    filtered_output = batch_data[class_mask, :]
    filtered_reid_idx = batch_reid_idx[class_mask]
    filtered_outputs.append(filtered_output)
    filtered_outputs_reid_idx.append(filtered_reid_idx)
print(yolo_outputs[0][0])
print(filtered_outputs_reid_idx[0].shape)
_, emb_vt2 = model2(xin, filtered_outputs, filtered_outputs_reid_idx)

print(len(emb_vt2))

cosine = np.dot(emb_vt1[1].detach().numpy(),emb_vt2[0].detach().numpy())/(norm(emb_vt1[0].detach().numpy())*norm(emb_vt2[0].detach().numpy()))
print(cosine)