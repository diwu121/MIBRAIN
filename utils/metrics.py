import torch
from numbers import Number


def eval_depth(pred, target):
    assert pred.shape == target.shape

    thresh = torch.max((target / pred), (pred / target))
    # print(len(thresh))

    d1 = torch.sum(thresh < 1.25).float() / len(thresh)
    d2 = torch.sum(thresh < 1.25 ** 2).float() / len(thresh)
    d3 = torch.sum(thresh < 1.25 ** 3).float() / len(thresh)

    diff = pred - target
    diff_log = torch.log(pred) - torch.log(target)

    abs_rel = torch.mean(torch.abs(diff) / target)
    sq_rel = torch.mean(torch.pow(diff, 2) / target)

    rmse = torch.sqrt(torch.mean(torch.pow(diff, 2)))
    rmse_log = torch.sqrt(torch.mean(torch.pow(diff_log, 2)))

    log10 = torch.mean(torch.abs(torch.log10(pred) - torch.log10(target)))
    silog = torch.sqrt(torch.pow(diff_log, 2).mean() -
                       0.5 * torch.pow(diff_log.mean(), 2))

    return {'d1': d1.item(), 'd2': d2.item(), 'd3': d3.item(), 'abs_rel': abs_rel.item(),
            'sq_rel': sq_rel.item(), 'rmse': rmse.item(), 'rmse_log': rmse_log.item(),
            'log10': log10.item(), 'silog': silog.item()}


def cropping_img(pred, gt_depth):
    min_depth_eval = 1e-3

    max_depth_eval = 10

    pred[torch.isinf(pred)] = max_depth_eval
    pred[torch.isnan(pred)] = min_depth_eval

    valid_mask = torch.logical_and(
        gt_depth > min_depth_eval, gt_depth < max_depth_eval)

    eval_mask = torch.zeros(valid_mask.shape).to(device=valid_mask.device)
    eval_mask[45:471, 41:601] = 1

    valid_mask = torch.logical_and(valid_mask, eval_mask)

    return pred[valid_mask], gt_depth[valid_mask]

def accuracy(pred, target, topk=1, thrs=0.):
    """Calculate accuracy according to the prediction and target.

    Args:
        pred (torch.Tensor): The model prediction.
        target (torch.Tensor): The target of each prediction
        topk (int | tuple[int]): If the predictions in ``topk``
            matches the target, the predictions will be regarded as
            correct ones. Defaults to 1.
        thrs (Number, optional): Predictions with scores under
            the thresholds are considered negative. Default to 0.

    Returns:
        torch.Tensor | list[torch.Tensor] | list[list[torch.Tensor]]: Accuracy
            - torch.Tensor: If both ``topk`` and ``thrs`` is a single value.
            - list[torch.Tensor]: If one of ``topk`` or ``thrs`` is a tuple.
            - list[list[torch.Tensor]]: If both ``topk`` and ``thrs`` is a \
              tuple. And the first dim is ``topk``, the second dim is ``thrs``.
    """
    assert isinstance(topk, (int, tuple)), \
        f'topk should be a number or tuple, but got {type(topk)}.'
    assert isinstance(thrs, Number), \
        f'thrs should be a number, but got {type(thrs)}.'
    if isinstance(topk, int):
        topk = (topk, )
        return_single = True
    else:
        return_single = False

    maxk = max(topk)
    pred_score, pred_label = pred.topk(maxk, dim=1)
    pred_label = pred_label.t()
    if target.dim() == 2 and target.size() == pred.size():  # one-hot target
        _, target = target.topk(1, dim=1)
    correct = pred_label.eq(target.view(1, -1).expand_as(pred_label))

    res = []
    for k in topk:
        if thrs > 0.:
            # Only prediction values larger than thr are counted as correct
            _correct = correct & (pred_score.t() > thrs)
            correct_k = _correct[:k].reshape(-1).float().sum(0, keepdim=True)
        else:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / max(pred.size(0), 1)))
    return res[0] if return_single else res
