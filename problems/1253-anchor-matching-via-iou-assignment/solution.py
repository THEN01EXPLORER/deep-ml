import numpy as np

def match_anchors(anchors, gt_boxes, pos_threshold=0.5, neg_threshold=0.4):
    """
    Assign each anchor a training label via IoU matching.

    Args:
        anchors: (N, 4) boxes in xyxy format [x1, y1, x2, y2]
        gt_boxes: (M, 4) ground-truth boxes in xyxy format
        pos_threshold: IoU >= this → positive (default 0.5)
        neg_threshold: IoU <  this → negative (default 0.4)

    Returns:
        labels:     (N,) int array with values {1=pos, 0=neg, -1=ignore}
        matched_gt: (N,) int array of matched GT index, or -1
    """

    anchors = np.asarray(anchors, dtype=float)
    gt_boxes = np.asarray(gt_boxes, dtype=float)

    N, M = len(anchors), len(gt_boxes)

    labels = np.full(N, -1, dtype=int)
    matched_gt = np.full(N, -1, dtype=int)

    if N == 0:
        return labels, matched_gt

    if M == 0:
        labels[:] = 0
        return labels, matched_gt

    # Calculate intersection
    x1 = np.maximum(anchors[:, None, 0], gt_boxes[None, :, 0])
    y1 = np.maximum(anchors[:, None, 1], gt_boxes[None, :, 1])
    x2 = np.minimum(anchors[:, None, 2], gt_boxes[None, :, 2])
    y2 = np.minimum(anchors[:, None, 3], gt_boxes[None, :, 3])

    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    # Calculate areas
    area_a = np.maximum(0, anchors[:, 2] - anchors[:, 0]) * \
             np.maximum(0, anchors[:, 3] - anchors[:, 1])
    area_g = np.maximum(0, gt_boxes[:, 2] - gt_boxes[:, 0]) * \
             np.maximum(0, gt_boxes[:, 3] - gt_boxes[:, 1])

    union = area_a[:, None] + area_g[None, :] - inter

    # Calculate IoU safely
    iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)

    # Best GT for each anchor
    best_gt = np.argmax(iou, axis=1)
    best_iou = np.max(iou, axis=1)

    # Assign labels based on thresholds
    labels[best_iou >= pos_threshold] = 1
    labels[best_iou < neg_threshold] = 0

    positive = labels == 1
    matched_gt[positive] = best_gt[positive]

    # Force every GT to match its best anchor
    for j in range(M):
        best_anchor = np.argmax(iou[:, j])
        labels[best_anchor] = 1
        matched_gt[best_anchor] = j

    return labels, matched_gt
