from models.yolox import YOLOX
import torch

import torch.nn.functional as F
model = YOLOX()


def bboxes_iou(bboxes_a, bboxes_b, xyxy=True):
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    if xyxy:
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
    else:
        tl = torch.max(
            (bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2),
        )
        br = torch.min(
            (bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2),
        )

        area_a = torch.prod(bboxes_a[:, 2:], 1)
        area_b = torch.prod(bboxes_b[:, 2:], 1)
    en = (tl < br).type(tl.type()).prod(dim=2)
    area_i = torch.prod(br - tl, 2) * en  # * ((tl < br).all())
    return area_i / (area_a[:, None] + area_b - area_i)

outputs = torch.tensor([
    [[100, 200, 300, 400, 0.9, 0.9, 1],
        [10.0, 20.0, 30.0, 40.0, 0.9, 0.8, 0],
        [15.0, 25.0, 35.0, 45.0, 0.7, 0.6, 0],
        [20.0, 30.0, 40.0, 50.0, 0.6, 0.5, 0],
        [25.0, 35.0, 45.0, 55.0, 0.5, 0.7, 0],
        [30.0, 40.0, 50.0, 60.0, 0.8, 0.9, 0],
        
    ],
    [
        [12.0, 22.0, 32.0, 42.0, 0.9, 0.7, 0],
        [17.0, 27.0, 37.0, 47.0, 0.8, 0.6, 0],
        [22.0, 32.0, 42.0, 52.0, 0.7, 0.5, 0],
        [27.0, 37.0, 47.0, 57.0, 0.6, 0.8, 0],
        [32.0, 42.0, 52.0, 62.0, 0.5, 0.9, 0],
        [100, 200, 300, 400, 0.6, 0.7, 0]
    ]
], dtype=torch.float32)

targets = torch.tensor([
    [   [14.0, 24.0, 34.0, 44.0, 0.7, 0.6, 0],
        [9.0, 19.0, 29.0, 39.0, 0.9, 0.8, 0],        
        [19.0, 29.0, 39.0, 49.0, 0.6, 0.5, 1],
        [24.0, 34.0, 44.0, 54.0, 0.5, 0.7, 0],
        [29.0, 39.0, 49.0, 59.0, 0.8, 0.9, 0]
    ],
    [
        [11.0, 21.0, 31.0, 41.0, 0.9, 0.7, 0],
        [16.0, 26.0, 36.0, 46.0, 0.8, 0.6, 0],
        [21.0, 31.0, 41.0, 51.0, 0.7, 0.5, 0],
        [26.0, 36.0, 46.0, 56.0, 0.6, 0.8, 0],
        [31.0, 41.0, 51.0, 61.0, 0.5, 0.9, 0],
    ]
], dtype=torch.float32)

def get_assignments(

        batch_idx,
        num_gt,
        gt_bboxes_per_image,
        gt_classes,
        bboxes_preds_per_image,
        cls_preds,
        obj_preds,
        mode="gpu",
    ):

        if mode == "cpu":
            print("-----------Using CPU for the Current Batch-------------")
            gt_bboxes_per_image = gt_bboxes_per_image.cpu().float()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu().float()
            gt_classes = gt_classes.cpu().float()

        cls_preds_ = cls_preds[batch_idx]
        obj_preds_ = obj_preds[batch_idx]
        num_in_boxes_anchor = bboxes_preds_per_image.shape[0]

        if mode == "cpu":
            gt_bboxes_per_image = gt_bboxes_per_image.cpu()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu()

        pair_wise_ious = bboxes_iou(gt_bboxes_per_image, bboxes_preds_per_image, False)
        gt_cls_per_image = (
            F.one_hot(gt_classes.to(torch.int64), 2)
            .float()
        )
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)

        if mode == "cpu":
            cls_preds_, obj_preds_ = cls_preds_.cpu(), obj_preds_.cpu()

        with torch.cuda.amp.autocast(enabled=False):
            #print(cls_preds_, obj_preds_)
            cls_preds_ = (
                cls_preds_.float().sigmoid_() * obj_preds_.float().sigmoid_()
            ).sqrt()   
            pair_wise_cls_loss = F.binary_cross_entropy(
                cls_preds_.unsqueeze(0).repeat(num_gt, 1, 1),
                gt_cls_per_image.unsqueeze(1).repeat(1, num_in_boxes_anchor, 1),
                reduction="none"
            ).sum(-1)
        del cls_preds_
        #print(pair_wise_cls_loss.shape, pair_wise_ious_loss.shape)
        cost = (
            pair_wise_cls_loss
            + 3.0 * pair_wise_ious_loss

        )

        (
            num_fg,
            gt_matched_classes,
            pred_ious_this_matching,
            matched_gt_inds,
        ) = simota_matching(cost, pair_wise_ious, gt_classes, num_gt)
        del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss

        if mode == "cpu":
            gt_matched_classes = gt_matched_classes.cuda()
            #fg_mask = fg_mask.cuda()
            pred_ious_this_matching = pred_ious_this_matching.cuda()
            matched_gt_inds = matched_gt_inds.cuda()

        return (
            gt_matched_classes,
            #fg_mask,
            pred_ious_this_matching,
            matched_gt_inds,
            num_fg,
        )

def simota_matching(cost, pair_wise_ious, gt_classes, num_gt):
        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)

        n_candidate_k = min(10, pair_wise_ious.size(1))
        topk_ious, _ = torch.topk(pair_wise_ious, n_candidate_k, dim=1)
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)

        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(
                cost[gt_idx], k=dynamic_ks[gt_idx], largest=False
            )
            matching_matrix[gt_idx][pos_idx] = 1

        del topk_ious, dynamic_ks, pos_idx

        anchor_matching_gt = matching_matrix.sum(0)
        # deal with the case that one anchor matches multiple ground-truths
        if anchor_matching_gt.max() > 1:
            multiple_match_mask = anchor_matching_gt > 1
            _, cost_argmin = torch.min(cost[:, multiple_match_mask], dim=0)
            matching_matrix[:, multiple_match_mask] *= 0
            matching_matrix[cost_argmin, multiple_match_mask] = 1
        fg_mask_inboxes = anchor_matching_gt > 0
        num_fg = fg_mask_inboxes.sum().item()


        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        gt_matched_classes = gt_classes[matched_gt_inds]

        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[
            fg_mask_inboxes
        ]
        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds
def matching(outputs, targets):
        #outputs shape: b, num_p, 7
        #matching box cua yolo detect voi box ground truth
        bbox_preds = outputs[:, :, :4]
        obj_preds = outputs[:, :, 4:5]
        cls_preds = outputs[:, :, 5:]
        batch_size = outputs.shape[0]
        for batch_idx in range(batch_size):
            num_gt = targets.shape[1]
            num_pred = outputs.shape[1]
            x_shift = torch.arange(num_pred).unsqueeze(0)
            y_shift = torch.arange(num_pred).unsqueeze(0)
            gt_bboxes_per_image = targets[batch_idx, :num_gt, 0:4]
            gt_classes = targets[batch_idx, :num_gt, 6]
            #gt_ids = targets[batch_idx, :num_gt, ]
            bboxes_preds_per_image = bbox_preds[batch_idx]
            expanded_strides = torch.ones_like(x_shift)
            (
                gt_matched_classes,
                # fg_mask,
                pred_ious_this_matching,
                matched_gt_inds,
                num_fg_img,
            ) = get_assignments(
                
                batch_idx,
                num_gt,
                gt_bboxes_per_image,
                gt_classes,
                bboxes_preds_per_image,
                cls_preds,
                obj_preds,
            )
            #print(fg_mask)
            print(matched_gt_inds)
matching(outputs, targets)
#model.matching(outputs, targets)
