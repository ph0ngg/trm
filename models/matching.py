import torch

import torch.nn.functional as F
from models.utils.boxes import bboxes_iou

def get_assignments(

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

        cls_preds_ = cls_preds
        obj_preds_ = obj_preds
        num_in_boxes_anchor = bboxes_preds_per_image.shape[0]

        if mode == "cpu":
            gt_bboxes_per_image = gt_bboxes_per_image.cpu()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu()

        pair_wise_ious = bboxes_iou(gt_bboxes_per_image, bboxes_preds_per_image, False)
        gt_cls_per_image = (
            F.one_hot(gt_classes.to(torch.int64), 1)
            .float()
        )
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)

        if mode == "cpu":
            cls_preds_, obj_preds_ = cls_preds_.cpu(), obj_preds_.cpu()

        with torch.cuda.amp.autocast(enabled=False):

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
        #outputs shape: num_p, 7
        #targets shape: num_p_ground truth, 6
        #matching box cua yolo detect voi box ground truth
        bbox_preds = outputs[:, :4]
        for k in range(len(bbox_preds)):
            bbox_preds[k][0] /= 1088
            bbox_preds[k][1] /= 608
            #print(bbox_preds[k][2])
            bbox_preds[k][2] /= 1088
            bbox_preds[k][3] /= 608
            x1, y1, x2, y2 = bbox_preds[k][:]
            x = (x1 + x2)/2
            y = (y1 + y2)/2
            w = x2 - x1
            h = y2 - y1
            bbox_preds[k][0] = x
            bbox_preds[k][1] = y
            bbox_preds[k][2] = w
            bbox_preds[k][3] = h
        obj_preds = outputs[:, 4:5]
        cls_preds = outputs[:, 6:7]

        # for batch_idx in range(batch_size):
        num_gt = targets.shape[0]
        num_pred = outputs.shape[0]
        gt_bboxes_per_image = targets[:num_gt, 2:6]
        gt_classes = targets[:num_gt, 0]
        #gt_ids = targets[batch_idx, :num_gt, ]
        bboxes_preds_per_image = bbox_preds
        (
            gt_matched_classes,
            # fg_mask,
            pred_ious_this_matching,
            matched_gt_inds,
            num_fg_img,
        ) = get_assignments(
            
            num_gt,
            gt_bboxes_per_image,
            gt_classes,
            bboxes_preds_per_image,
            cls_preds,
            obj_preds,
        )
        return gt_matched_classes, matched_gt_inds, num_fg_img