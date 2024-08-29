import torch
import cv2

from numpy.linalg import norm

from models.yolox import *

def CS(a, b):
    return np.dot(a.detach().numpy(),b.detach().numpy())/(norm(a.detach().numpy())*norm(b.detach().numpy()))


def xyxy2cxcywh(bboxes):
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    bboxes[:, 0] = bboxes[:, 0] + bboxes[:, 2] * 0.5
    bboxes[:, 1] = bboxes[:, 1] + bboxes[:, 3] * 0.5
    return bboxes

model1 = YOLOX()
model2 = Head2()

model1.load_state_dict(torch.load('/mnt/data_ubuntu/phongnn/yolox_m.pth')['model'], strict = False)
model2.load_state_dict(torch.load('/mnt/data_ubuntu/phongnn/kaggl/weights_epoch_20.pt')['model'])

path_to_img1 = '/mnt/data_ubuntu/phongnn/Market-1501-v15.09.15/bounding_box_train/0002_c1s1_000451_03.jpg'

# path_to_img2 = '/mnt/data_ubuntu/phongnn/Market-1501-v15.09.15/bounding_box_train/1496_c6s3_094292_04.jpg'

model1.eval()
model2.eval()

import os
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

query_path = '/mnt/data_ubuntu/phongnn/images/query'
gallery_path = '/mnt/data_ubuntu/phongnn/images/gallery'

query_vector_matrix = []
gallery_vector_matrix = []

query_id_list = []
gallery_id_list = []

true = []
false = []

for query_ids in tqdm(sorted(os.listdir(query_path))):
    query_img = cv2.imread(os.path.join(query_path, query_ids))
    query_img = cv2.resize(query_img, (64, 128))
    query_img = np.transpose(query_img, (2, 0, 1))
    query_img = np.expand_dims(query_img, 0)
    query_img = torch.Tensor(query_img)
    query_id = query_ids.split('_')[0]
    xin, yolo_outputs, reid_idx = model1(query_img)
    filtered_outputs = []
    filtered_outputs_reid_idx = []
    if len(yolo_outputs) == 0 or yolo_outputs[0] == None:
        continue
    for batch_idx in range(len(yolo_outputs)):
        batch_data = yolo_outputs[batch_idx]
        batch_reid_idx = reid_idx[batch_idx]
        #try:
        class_mask = batch_data[:, 6] == 0
        # except:
        #     print((yolo_outputs))
        filtered_output = batch_data[class_mask, :]
        filtered_reid_idx = batch_reid_idx[class_mask]
        filtered_outputs.append(filtered_output)
        filtered_outputs_reid_idx.append(filtered_reid_idx)
 #   _, query_vector = model2(xin, filtered_outputs, filtered_outputs_reid_idx)
 
    if len(filtered_outputs[0] > 1):
        conf_scores = filtered_outputs[0][:, 5].detach().numpy()
        highest_conf_idx = np.argmax(conf_scores)
        bboxs = filtered_outputs[0][:, :4].detach().numpy()
        bboxs[:, 0] /= 64 #x1, y1, x2, y2
        bboxs[:, 1] /= 128
        bboxs[:, 2] /= 64
        bboxs[:, 3] /= 128
        bboxs = xyxy2cxcywh(bboxs)
        #shape: n_peo, 4
        #--> batch, n_peo, 6
        dump_array = np.full((len(bboxs), 2), [0, -1])
        new_boxes = np.hstack((dump_array, bboxs))
        new_boxes = np.expand_dims(new_boxes, axis= 0)
        new_boxes = torch.from_numpy(new_boxes).float()
        query_vectors = model2(xin, 128, 64, new_boxes)
        final_query_vector = query_vectors[highest_conf_idx].detach().numpy()
        #print(final_query_vector)
        query_vector_matrix.append(final_query_vector)
        query_id_list.append(query_id)
    elif len(filtered_outputs[0] == 1):
        #try:
        bboxs = filtered_outputs[0][:, :4].detach().numpy()
        dump_array = np.full((len(bboxs), 2), [0, -1])
        new_boxes = np.hstack((dump_array, bboxs))
        new_boxes = np.expand_dims(new_boxes, axis= 0)
        query_vectors = model2(xin, 128, 64, new_boxes)
        final_query_vector = query_vectors[0].detach().numpy()
        query_vector_matrix.append(final_query_vector)
        query_id_list.append(query_id)
    else:
        continue
        # except:
        #     print(filtered_outputs[0])
    

for gallery_ids in tqdm(sorted(os.listdir(gallery_path))):
    gallery_id = gallery_ids.split('_')[0]
    if gallery_id == '-1':
        continue
    gallery_img = cv2.imread(os.path.join(gallery_path, gallery_ids))
    gallery_img = cv2.resize(gallery_img, (64, 128))
    gallery_img = np.transpose(gallery_img, (2, 0, 1))
    gallery_img = np.expand_dims(gallery_img, 0)
    gallery_img = torch.Tensor(gallery_img)

    xin, yolo_outputs, reid_idx = model1(gallery_img)
    filtered_outputs = []
    filtered_outputs_reid_idx = []
    if len(yolo_outputs) == 0 or yolo_outputs[0] == None:
        continue
    for batch_idx in range(len(yolo_outputs)):
        batch_data = yolo_outputs[batch_idx]
        batch_reid_idx = reid_idx[batch_idx]
        #try:
        class_mask = batch_data[:, 6] == 0
        # except:
        #     print((yolo_outputs))
        filtered_output = batch_data[class_mask, :]
        filtered_reid_idx = batch_reid_idx[class_mask]
        filtered_outputs.append(filtered_output)
        filtered_outputs_reid_idx.append(filtered_reid_idx)
 #   _, query_vector = model2(xin, filtered_outputs, filtered_outputs_reid_idx)
 
    if len(filtered_outputs[0] > 1):
        conf_scores = filtered_outputs[0][:, 5].detach().numpy()
        highest_conf_idx = np.argmax(conf_scores)
        bboxs = filtered_outputs[0][:, :4].detach().numpy()
        bboxs[:, 0] /= 64 #x1, y1, x2, y2
        bboxs[:, 1] /= 128
        bboxs[:, 2] /= 64
        bboxs[:, 3] /= 128
        bboxs = xyxy2cxcywh(bboxs)
        #shape: n_peo, 4
        #--> batch, n_peo, 6
        dump_array = np.full((len(bboxs), 2), [0, -1])
        new_boxes = np.hstack((dump_array, bboxs))
        new_boxes = np.expand_dims(new_boxes, axis= 0)
        new_boxes = torch.from_numpy(new_boxes).float()
        gallery_vectors = model2(xin, 128, 64, new_boxes)
        final_gallery_vector = gallery_vectors[highest_conf_idx].detach().numpy()
        gallery_vector_matrix.append(final_gallery_vector)
        gallery_id_list.append(gallery_id)
    elif len(filtered_outputs[0] == 1):
        #try:
        bboxs = filtered_outputs[0][:, :4].detach().numpy()
        dump_array = np.full((len(bboxs), 2), [0, -1])
        new_boxes = np.hstack((dump_array, bboxs))
        new_boxes = np.expand_dims(new_boxes, axis= 0)
        gallery_vectors = model2(xin, 128, 64, new_boxes)
        final_gallery_vector = gallery_vectors[0].detach().numpy()
        gallery_vector_matrix.append(final_gallery_vector)
        gallery_id_list.append(gallery_id)
    else:
        continue


cosine_matrix = cosine_similarity(query_vector_matrix, gallery_vector_matrix)
print(max(max(row) for row in np.array(cosine_matrix)))

true = []
false = []
print(cosine_matrix.shape)

for i in range(len(query_id_list)):
    for j in range(len(gallery_id_list)):
        if query_id_list[i] == gallery_id_list[j]:
            true.append(cosine_matrix[i, j])
        else:
            false.append(cosine_matrix[i, j])
print(max(true))
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(10, 6))

# Vẽ histogram
sns.histplot(true, bins=30, kde=False, color='green', label='Histogram', stat= 'probability', alpha= 0.6)
sns.histplot(false, bins=30, kde=False, color='red', label='Histogram',  stat= 'probability', alpha= 0.6)


# Vẽ đường KDE
# sns.kdeplot(true, color='green', label='true')
# sns.kdeplot(false, color='red', label = 'false')

# Thêm tiêu đề và nhãn cho các trục
plt.title('Epoch 40')
plt.xlabel('Value')
plt.ylabel('Frequency')

# Thêm chú thích
plt.legend()

# Hiển thị đồ thị
plt.show()