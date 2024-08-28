import torch
import cv2

from numpy.linalg import norm

from models.yolox1 import *

def CS(a, b):
    return np.dot(a.detach().numpy(),b.detach().numpy())/(norm(a.detach().numpy())*norm(b.detach().numpy()))

model1 = YOLOX()
model2 = Head2()

model1.load_state_dict(torch.load('/mnt/data_ubuntu/phongnn/kaggl/Towards-Realtime-MOT/yolox_l.pth')['model'], strict = False)
model2.load_state_dict(torch.load('/mnt/data_ubuntu/phongnn/latest_2_loss.pt')['model'])

path_to_img1 = '/mnt/data_ubuntu/phongnn/Market-1501-v15.09.15/bounding_box_train/0002_c1s1_000451_03.jpg'

# path_to_img2 = '/mnt/data_ubuntu/phongnn/Market-1501-v15.09.15/bounding_box_train/1496_c6s3_094292_04.jpg'

model1.eval()
model2.eval()

img1 = cv2.imread(path_to_img1)
#img2 = cv2.imread(path_to_img2)
img1 = cv2.resize(img1, (1088, 608))
img1 = np.transpose(img1, (2, 0, 1))

img1 = np.expand_dims(img1, 0)
img1 = torch.Tensor(img1)

#img2 = cv2.resize(img2, (1088, 608))
# img2 = np.transpose(img2, (2, 0, 1))

# img2 = np.expand_dims(img2, 0)
# img2 = torch.Tensor(img2)

#xin, yolo_outputs, reid_idx = model1(img1)
# print(yolo_outputs)
# filtered_outputs = []
# filtered_outputs_reid_idx = []
# for batch_idx in range(len(yolo_outputs)):
#     batch_data = yolo_outputs[batch_idx]
#     batch_reid_idx = reid_idx[batch_idx]
#     class_mask = batch_data[:, 6] == 0
#     filtered_output = batch_data[class_mask, :]
#     filtered_reid_idx = batch_reid_idx[class_mask]
#     filtered_outputs.append(filtered_output)
#     filtered_outputs_reid_idx.append(filtered_reid_idx)
# print(filtered_outputs)
# _, emb_vt1 = model2(xin, filtered_outputs, filtered_outputs_reid_idx)
# print(len(emb_vt1))
# for i, person in enumerate(filtered_outputs[0]):
#     x1, y1, x2, y2 = person[0:4]
#     person_img = img1[0].detach().numpy()
#     person_img = np.transpose(person_img, (1, 2, 0))
#     x1 = int(x1)
#     y1 = int(y1)
#     x2 = int(x2)
#     y2 = int(y2)
#     person_img = person_img[y1:y2, x1:x2, :]
#     cv2.imwrite(f'images/img1_{i}.jpg', person_img)
# print(filtered_outputs_reid_idx[0].shape)

# xin, yolo_outputs, reid_idx = model1(img2)
# filtered_outputs = []
# filtered_outputs_reid_idx = []
# for batch_idx in range(len(yolo_outputs)):
#     batch_data = yolo_outputs[batch_idx]
#     batch_reid_idx = reid_idx[batch_idx]
#     class_mask = batch_data[:, 6] == 0
#     filtered_output = batch_data[class_mask, :]
#     filtered_reid_idx = batch_reid_idx[class_mask]
#     filtered_outputs.append(filtered_output)
#     filtered_outputs_reid_idx.append(filtered_reid_idx)
# for i, person in enumerate(filtered_outputs[0]):
#     x1, y1, x2, y2 = person[0:4]
#     person_img = img2[0].detach().numpy()
#     person_img = np.transpose(person_img, (1, 2, 0))
#     x1 = int(x1)
#     y1 = int(y1)
#     x2 = int(x2)
#     y2 = int(y2)
#     person_img = person_img[y1:y2, x1:x2, :]
#     cv2.imwrite(f'images/img2_{i}.jpg', person_img)
# print(filtered_outputs_reid_idx[0].shape)
# _, emb_vt2 = model2(xin, filtered_outputs, filtered_outputs_reid_idx)

# print(len(emb_vt2[0]))
# a = np.zeros((len(emb_vt1), len(emb_vt2)))
# for i in range(len(emb_vt1)):
#     for j in range(len(emb_vt2)):
#         a[i, j] = CS(emb_vt1[i], emb_vt2[j])
#         print(f'a_{i}_{j} = {a[i, j]}')

# weights = torch.load('/mnt/data_ubuntu/phongnn/Towards-Realtime-MOT/weights/run22_08_06_22/latest.pt')
# print(weights['epoch'])



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

    _, query_vector = model2(xin, filtered_outputs, filtered_outputs_reid_idx)

    if len(filtered_outputs[0] > 1):
        conf_scores = filtered_outputs[0][:, 5].detach().numpy()
        highest_conf_idx = np.argmax(conf_scores)
        final_query_vector = query_vector[highest_conf_idx].detach().numpy()
        query_vector_matrix.append(final_query_vector)
        query_id_list.append(query_id)
    elif len(filtered_outputs[0] == 1):
        #try:
        final_query_vector = query_vector[0].detach().numpy()
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
        class_mask = batch_data[:, 6] == 0
        filtered_output = batch_data[class_mask, :]
        filtered_reid_idx = batch_reid_idx[class_mask]
        filtered_outputs.append(filtered_output)
        filtered_outputs_reid_idx.append(filtered_reid_idx)
    _, gallery_vector = model2(xin, filtered_outputs, filtered_outputs_reid_idx)
    if len(filtered_outputs[0] > 1):
        conf_scores = filtered_outputs[0][:, 5].detach().numpy()
        highest_conf_idx = np.argmax(conf_scores)
        final_gallery_vector = gallery_vector[highest_conf_idx].detach().numpy()
        gallery_vector_matrix.append(final_gallery_vector)
        gallery_id_list.append(gallery_id)
    elif len(filtered_outputs[0] == 1):
        final_gallery_vector = gallery_vector[0].detach().numpy()
        gallery_vector_matrix.append(final_gallery_vector)
        gallery_id_list.append(gallery_id)
    else:
        continue


# for query_ids in tqdm(sorted(os.listdir(query_path))):

#     query_img = cv2.imread(os.path.join(query_path, query_ids))
#     query_img = np.transpose(query_img, (2, 0, 1))
#     query_img = np.expand_dims(query_img, 0)
#     query_img = torch.Tensor(query_img)
#     query_id = query_ids.split('_')[0]
#     xin, yolo_outputs, reid_idx = model1(query_img)
    

#     if len(yolo_outputs) == 0 or yolo_outputs[0] == None:
#         continue
#     for batch_idx in range(len(yolo_outputs)):
#         batch_data = yolo_outputs[batch_idx]
#         #try:
#         class_mask = batch_data[:, 6] == 0
#         # except:
#         #     print((yolo_outputs))
#         filtered_output = batch_data[class_mask, :]
#         print(filtered_output.shape)
#     #_, query_vector = model2(xin, filtered_outputs, filtered_outputs_reid_idx)
#     if filtered_output.shape[0] > 1:

#         conf_scores = filtered_output[:, 5].detach().numpy()

#         highest_conf_idx = np.argmax(conf_scores)
#         final_query_vector = filtered_output[highest_conf_idx].detach().numpy()
#         query_vector_matrix.append(final_query_vector[7:])
#         query_id_list.append(query_id)
#     elif filtered_output.shape[0] == 1:
#         #try:
#         final_query_vector = filtered_output[0].detach().numpy()
#         query_vector_matrix.append(final_query_vector[7:])
#         query_id_list.append(query_id)
#     else:
#         continue
#         # except:
#         #     print(filtered_outputs[0])
    

# for gallery_ids in tqdm(sorted(os.listdir(gallery_path))):
#     gallery_id = gallery_ids.split('_')[0]
#     if gallery_id == '-1':
#         continue
#     gallery_img = cv2.imread(os.path.join(gallery_path, gallery_ids))
#     gallery_img = np.transpose(gallery_img, (2, 0, 1))
#     gallery_img = np.expand_dims(gallery_img, 0)
#     gallery_img = torch.Tensor(gallery_img)

#     xin, yolo_outputs, reid_idx = model1(gallery_img)


#     if len(yolo_outputs) == 0 or yolo_outputs[0] == None:
#         continue
#     for batch_idx in range(len(yolo_outputs)):
#         batch_data = yolo_outputs[batch_idx]
#         #try:
#         class_mask = batch_data[:, 6] == 0
#         # except:
#         #     print((yolo_outputs))
#         filtered_output = batch_data[class_mask, :]
        

#     #_, gallery_vector = model2(xin, filtered_outputs, filtered_outputs_reid_idx)
#     if filtered_output.shape[0] > 1:

#         conf_scores = filtered_output[:, 5].detach().numpy()

#         highest_conf_idx = np.argmax(conf_scores)
#         final_gallery_vector = filtered_output[highest_conf_idx].detach().numpy()
#         gallery_vector_matrix.append(final_gallery_vector[7:])
#         gallery_id_list.append(gallery_id)
#     elif filtered_output.shape[0] == 1:
#         #try:
#         final_gallery_vector = filtered_output[0].detach().numpy()
#         gallery_vector_matrix.append(final_gallery_vector[7:])
#         gallery_id_list.append(gallery_id)
#     else:
#         continue


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
plt.title('Histogram and KDE Plot')
plt.xlabel('Value')
plt.ylabel('Frequency')

# Thêm chú thích
plt.legend()

# Hiển thị đồ thị
plt.show()