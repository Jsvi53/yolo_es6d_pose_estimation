import numpy as np


def get_bbox(label):
    img_length = label.shape[1]
    img_width = label.shape[0]
    rows = np.any(label, axis=1)
    cols = np.any(label, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    rmax += 1
    cmax += 1
    r_b = rmax - rmin
    c_b = cmax - cmin
    wid = max(r_b, c_b)
    extend_wid = int(wid / 8)
    center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
    rmin = center[0] - int(wid / 2) - extend_wid
    rmax = center[0] + int(wid / 2) + extend_wid
    cmin = center[1] - int(wid / 2) - extend_wid
    cmax = center[1] + int(wid / 2) + extend_wid
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > img_width:
        delt = rmax - img_width
        rmax = img_width
        rmin -= delt
    if cmax > img_length:
        delt = cmax - img_length
        cmax = img_length
        cmin -= delt
    return rmin, rmax, cmin, cmax
