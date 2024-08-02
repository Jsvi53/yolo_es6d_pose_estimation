import torch
import numpy as np
import math
import torch.nn.functional as F


def quaternion_matrix(quaternion):
    """Return homogeneous rotation matrix from quaternion.

    >>> M = quaternion_matrix([0.99810947, 0.06146124, 0, 0])
    >>> numpy.allclose(M, rotation_matrix(0.123, [1, 0, 0]))
    True
    >>> M = quaternion_matrix([1, 0, 0, 0])
    >>> numpy.allclose(M, numpy.identity(4))
    True
    >>> M = quaternion_matrix([0, 1, 0, 0])
    >>> numpy.allclose(M, numpy.diag([1, -1, -1, 1]))
    True

    """
    _EPS = np.finfo(float).eps * 4.0
    q = np.array(quaternion, dtype=np.float64, copy=True)
    n = np.dot(q, q)
    if n < _EPS:
        return np.identity(4)
    q *= math.sqrt(2.0 / n)
    q = np.outer(q, q)
    return np.array([[1.0 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0], 0.0],
                    [q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0], 0.0],
                    [q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2], 0.0],
                    [0.0, 0.0, 0.0, 1.0],])


def pose_PostProcessing(results, resized_xyz):
    """
    get final transform matrix T=[R|t] from prediction results
    :return: T[3, 4]
    """
    preds = dict()
    preds["cls_id"] = torch.from_numpy(np.array(results[0])).contiguous()
    preds["pred_r"] = torch.from_numpy(np.array(results[1])).view((4, 32, 32)).unsqueeze(0).contiguous()
    preds["pred_s"] = torch.from_numpy(np.array(results[2])).view((32, 32)).unsqueeze(0).contiguous()
    preds["pred_t"] = torch.from_numpy(np.array(results[3])).view((3, 32, 32)).unsqueeze(0).contiguous()
    preds["xyz"] = torch.tensor(resized_xyz).unsqueeze(0)  # pred['xyz']:(1, 3, 128, 128)
    b, c, h, w = preds["pred_r"].size()
    px = preds["pred_r"].view(b, 4, -1)
    pt = preds["pred_t"].view(b, 3, -1)
    ps = preds["pred_s"].view(b, -1)
    mask = preds["xyz"][:, 0].unsqueeze(dim=1)
    mask = F.interpolate(mask, size=(h, w)).squeeze(dim=1).view(b, -1)
    res_T = []
    for i in range(b):
        valid_pixels = mask[i].nonzero().view(-1)
        num_val = valid_pixels.size()[0]
        if num_val < 32:
            valid_pixels = torch.ones(mask[i].size()).nonzero().view(-1)
            num_val = valid_pixels.size()[0]
        q = px[i].view(4, -1)[:, valid_pixels]
        t = pt[i].view(3, -1)[:, valid_pixels]
        s = ps[i].view(-1)[valid_pixels]
        s_id = torch.argmax(s)
        _q = q[:, s_id].numpy()
        _r = quaternion_matrix(_q)[:3, :3]
        _r = torch.from_numpy(_r).float()
        _t = t[:, s_id].view(3, 1)
        res_T.append(torch.cat([_r, _t], dim=1).numpy())

    return res_T[0]
