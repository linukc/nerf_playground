""" Metrics for the project. """

import torch

#pylint: disable=too-many-locals
def calc_depth_metrics(pred_t: torch.Tensor, gt_t: torch.Tensor, masks:torch.Tensor=None) -> dict:
    """
    Calc depth metrics.
    
    Arguments
    ---------
    pred_t: [b, h, w]
        Predicted tensor with depth values.
    gt_t: [b, h, w]
        GT depth tensor.
    masks: [b, h, w]
        Mask tensor. Define where to calculate metrics.

    Returns
    -------
    metrics:
        Dict with scale, abs_diff, abs_rel, sq_rel, a_1, a_2, a_3 metrics.
    """

    batch_size = gt_t.size(0)
    scale, abs_diff, abs_rel, sq_rel, a_1, a_2, a_3 = 0, 0, 0, 0, 0, 0, 0

    for pair in zip(pred_t, gt_t, masks):
        pred, grount_truth, one_mask = pair

        valid_gt = grount_truth[one_mask]
        valid_pred = pred[one_mask]
        scale += torch.median(valid_gt) / torch.median(valid_pred)

        thresh = torch.max((valid_gt / valid_pred), (valid_pred / valid_gt))
        a_1 += (thresh < 1.25).float().mean()
        a_2 += (thresh < 1.25 ** 2).float().mean()
        a_3 += (thresh < 1.25 ** 3).float().mean()

        abs_diff += torch.mean(torch.abs(valid_gt - valid_pred))
        abs_rel += torch.mean(torch.abs(valid_gt - valid_pred) / valid_gt)
        sq_rel += torch.mean(((valid_gt - valid_pred)**2) / valid_gt)

        names = ["scale", "abs_diff", "abs_rel", "sq_rel", "a1", "a2", "a3"]
        metrics = [round(metric.item() / batch_size, 3)  for metric in
            [scale, abs_diff, abs_rel, sq_rel, a_1, a_2, a_3]]

    return {f"median_{k}": v for k, v in zip(names, metrics)}
