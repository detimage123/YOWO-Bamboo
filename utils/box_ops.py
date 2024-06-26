import torch
import numpy as np


def get_ious(bboxes1,
             bboxes2,
             box_mode="xyxy",
             iou_type="iou"):
    """
    Compute iou loss of type ['iou', 'giou', 'linear_iou']

    Args:
        inputs (tensor): pred values
        targets (tensor): target values
        weight (tensor): loss weight
        box_mode (str): 'xyxy' or 'ltrb', 'ltrb' is currently supported.
        loss_type (str): 'giou' or 'iou' or 'linear_iou'
        reduction (str): reduction manner

    Returns:
        loss (tensor): computed iou loss.
    """
    if box_mode == "ltrb":
        bboxes1 = torch.cat((-bboxes1[..., :2], bboxes1[..., 2:]), dim=-1)
        bboxes2 = torch.cat((-bboxes2[..., :2], bboxes2[..., 2:]), dim=-1)
    elif box_mode != "xyxy":
        raise NotImplementedError

    eps = torch.finfo(torch.float32).eps

    bboxes1_area = (bboxes1[..., 2] - bboxes1[..., 0]).clamp_(min=0) \
        * (bboxes1[..., 3] - bboxes1[..., 1]).clamp_(min=0)
    bboxes2_area = (bboxes2[..., 2] - bboxes2[..., 0]).clamp_(min=0) \
        * (bboxes2[..., 3] - bboxes2[..., 1]).clamp_(min=0)

    w_intersect = (torch.min(bboxes1[..., 2], bboxes2[..., 2])
                   - torch.max(bboxes1[..., 0], bboxes2[..., 0])).clamp_(min=0)
    h_intersect = (torch.min(bboxes1[..., 3], bboxes2[..., 3])
                   - torch.max(bboxes1[..., 1], bboxes2[..., 1])).clamp_(min=0)

    area_intersect = w_intersect * h_intersect
    area_union = bboxes2_area + bboxes1_area - area_intersect
    ious = area_intersect / area_union.clamp(min=eps)

    if iou_type == "iou":
        return ious
    elif iou_type == "giou":
        g_w_intersect = torch.max(bboxes1[..., 2], bboxes2[..., 2]) \
            - torch.min(bboxes1[..., 0], bboxes2[..., 0])
        g_h_intersect = torch.max(bboxes1[..., 3], bboxes2[..., 3]) \
            - torch.min(bboxes1[..., 1], bboxes2[..., 1])
        ac_uion = g_w_intersect * g_h_intersect
        gious = ious - (ac_uion - area_union) / ac_uion.clamp(min=eps)
        return gious
    elif iou_type == "diou":
        cx1 = (bboxes1[..., 0] + bboxes1[..., 2]) / 2
        cy1 = (bboxes1[..., 1] + bboxes1[..., 3]) / 2
        cx2 = (bboxes2[..., 0] + bboxes2[..., 2]) / 2
        cy2 = (bboxes2[..., 1] + bboxes2[..., 3]) / 2
        rho = ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5
        w1 = bboxes1[..., 2] - bboxes1[..., 0]
        h1 = bboxes1[..., 3] - bboxes1[..., 1]
        w2 = bboxes2[..., 2] - bboxes2[..., 0]
        h2 = bboxes2[..., 3] - bboxes2[..., 1]
        c2 = ((w1 + w2) ** 2 + (h1 + h2) ** 2) / 4
        dious = ious - (rho ** 2) / c2.clamp(min=eps)
        return dious
    else:
        raise NotImplementedError


def rescale_bboxes(bboxes, orig_size):
    orig_w, orig_h = orig_size[0], orig_size[1]
    bboxes[..., [0, 2]] = np.clip(
        bboxes[..., [0, 2]] * orig_w, a_min=0., a_max=orig_w
        )
    bboxes[..., [1, 3]] = np.clip(
        bboxes[..., [1, 3]] * orig_h, a_min=0., a_max=orig_h
        )
    
    return bboxes



if __name__ == '__main__':
    box1 = torch.tensor([[10, 10, 20, 20]])
    box2 = torch.tensor([[15, 15, 25, 25]])
