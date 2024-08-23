import torch
from scipy.optimize import linear_sum_assignment

def calculate_iou(boxes1, boxes2):
    N = boxes1.size(0)
    M = boxes2.size(0)
    
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    
    wh = torch.clamp(rb - lt, min=0)
    inter_area = wh[:, :, 0] * wh[:, :, 1]
    
    union_area = area1[:, None] + area2 - inter_area
    iou = inter_area / union_area
    return iou

def match_boxes(pred_boxes, gt_boxes, iou_threshold=0.1):
    iou = calculate_iou(pred_boxes, gt_boxes)
    
    # Chuyển đổi IoU thành ma trận chi phí (chi phí = 1 - IoU)
    cost_matrix = 1 - iou.numpy()
    
    # Áp dụng thuật toán Hungarian để gán
    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    
    # Lọc các cặp gán dựa trên ngưỡng IoU
    matches = []
    for row, col in zip(row_indices, col_indices):
        if iou[row, col] > iou_threshold:
            matches.append((row, col, iou[row, col].item()))
    
    return matches

# Ví dụ về các hộp dự đoán và ground truth
pred_boxes = torch.tensor([[10, 10, 50, 50], [20, 20, 60, 60]], dtype=torch.float)
gt_boxes = torch.tensor([[15, 15, 45, 45], [30, 30, 70, 70], [5, 5, 20, 20]], dtype=torch.float)

# Gán các hộp dự đoán với hộp ground truth
matches = match_boxes(pred_boxes, gt_boxes)
print(matches)
print("Các cặp gán giữa hộp dự đoán và hộp ground truth:")
for match in matches:
    print(f"Hộp dự đoán {match[0]} khớp với hộp ground truth {match[1]} với IoU {match[2]}")