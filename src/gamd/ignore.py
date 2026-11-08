"""Ignore regions for torchvision Mask R-CNN training.

Targets may carry `ignore` (H×W bool, image coordinates). Anchors (RPN) and
proposals (box head) that would be sampled as *background* are dropped when
their centre lies in an ignore region, so unlabelled mounds at partly
inventoried sites (Etowah B/C, Kolomoki's other mounds, ...) are never taught
as "not a mound". Positives are unaffected. Requires images that the model's
transform does not resize (our chips are 200 px with min_size=max_size=200).
"""


def _drop_background(labels, boxes, ig):
    if ig is None or not bool(ig.any()) or boxes.numel() == 0:
        return labels
    H, W = ig.shape
    cx = ((boxes[:, 0] + boxes[:, 2]) / 2).long().clamp(0, W - 1)
    cy = ((boxes[:, 1] + boxes[:, 3]) / 2).long().clamp(0, H - 1)
    labels = labels.clone()
    labels[(labels == 0) & ig[cy, cx]] = -1          # -1 = neither positive nor negative to the samplers
    return labels


def enable_ignore(model):
    rpn, roi = model.rpn, model.roi_heads
    rpn_assign = rpn.assign_targets_to_anchors

    def assign_anchors(anchors, targets):
        labels, matched = rpn_assign(anchors, targets)
        return [_drop_background(l, a, t.get("ignore")) for l, a, t in zip(labels, anchors, targets)], matched
    rpn.assign_targets_to_anchors = assign_anchors

    roi_select, roi_assign = roi.select_training_samples, roi.assign_targets_to_proposals
    state = {}

    def select(proposals, targets):
        state["ig"] = [t.get("ignore") for t in targets]
        return roi_select(proposals, targets)

    def assign_proposals(proposals, gt_boxes, gt_labels):
        matched, labels = roi_assign(proposals, gt_boxes, gt_labels)
        return matched, [_drop_background(l, p, ig) for l, p, ig in zip(labels, proposals, state["ig"])]
    roi.select_training_samples, roi.assign_targets_to_proposals = select, assign_proposals
    return model
