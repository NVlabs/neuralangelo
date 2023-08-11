'''
-----------------------------------------------------------------------------
Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
-----------------------------------------------------------------------------
'''

import torch


SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
]
SH_C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
]
SH_C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]


def get_spherical_harmonics(dirs, levels):
    # Evaluate spherical harmonics bases at unit directions, without taking linear combination.
    vals = torch.empty((*dirs.shape[:-1], (levels + 1) ** 2), device=dirs.device)
    vals[..., 0] = SH_C0
    if levels >= 1:
        x, y, z = dirs.unbind(-1)
        vals[..., 1] = -SH_C1 * y
        vals[..., 2] = SH_C1 * z
        vals[..., 3] = -SH_C1 * x
    if levels >= 2:
        xx, yy, zz = x * x, y * y, z * z
        xy, yz, xz = x * y, y * z, x * z
        vals[..., 4] = SH_C2[0] * xy
        vals[..., 5] = SH_C2[1] * yz
        vals[..., 6] = SH_C2[2] * (2.0 * zz - xx - yy)
        vals[..., 7] = SH_C2[3] * xz
        vals[..., 8] = SH_C2[4] * (xx - yy)
    if levels >= 3:
        vals[..., 9] = SH_C3[0] * y * (3 * xx - yy)
        vals[..., 10] = SH_C3[1] * xy * z
        vals[..., 11] = SH_C3[2] * y * (4 * zz - xx - yy)
        vals[..., 12] = SH_C3[3] * z * (2 * zz - 3 * xx - 3 * yy)
        vals[..., 13] = SH_C3[4] * x * (4 * zz - xx - yy)
        vals[..., 14] = SH_C3[5] * z * (xx - yy)
        vals[..., 15] = SH_C3[6] * x * (xx - 3 * yy)
    if levels >= 4:
        vals[..., 16] = SH_C4[0] * xy * (xx - yy)
        vals[..., 17] = SH_C4[1] * yz * (3 * xx - yy)
        vals[..., 18] = SH_C4[2] * xy * (7 * zz - 1)
        vals[..., 19] = SH_C4[3] * yz * (7 * zz - 3)
        vals[..., 20] = SH_C4[4] * (zz * (35 * zz - 30) + 3)
        vals[..., 21] = SH_C4[5] * xz * (7 * zz - 3)
        vals[..., 22] = SH_C4[6] * (xx - yy) * (7 * zz - 1)
        vals[..., 23] = SH_C4[7] * xz * (xx - 3 * yy)
        vals[..., 24] = SH_C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy))
    if levels >= 5:
        raise NotImplementedError
    return vals
