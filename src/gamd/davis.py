"""Load Davis et al. (2021) shell-ring/mound Mask R-CNN without ArcGIS.

Source: Davis, Caspari, Lipo & Sanger (2021), "Deep learning reveals extent of
Archaic Native American shell-ring building practices", J. Archaeol. Sci. 132:105433.
Model + training data: https://doi.org/10.26207/ry6k-q463 (CC BY 4.0).

The .pth inside his ArcGIS .dlpk is a legacy (pre-1.6) torch pickle holding
{'model': state_dict, 'opt': optimizer state}. The state dict matches torchvision's
maskrcnn_resnet50_fpn exactly (494 tensors). We load it with weights_only=True so
only tensors and an explicit allowlist of harmless types can be unpickled; any
other code reference in the file makes the load fail instead of executing.
"""
import json
from pathlib import Path

import numpy as np
import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn

# Class ids as stored in the model's .emd (0 = background).
CLASSES = {1: "Modern", 2: "Mound", 3: "Ring"}

# The only non-tensor globals the file references (verified with pickletools):
# numpy scalar + dtype (pickled by numpy 1.x under numpy.core.*), codecs.encode
# (how numpy 1.x pickled raw bytes), OrderedDict and torch tensor rebuilders
# (allowed by torch already). Names are given explicitly because numpy 2 moved
# numpy.core -> numpy._core, so the objects' own qualified names no longer match.
import codecs
try:
    from numpy._core.multiarray import scalar as _np_scalar   # numpy >= 2
except ImportError:
    from numpy.core.multiarray import scalar as _np_scalar    # numpy 1.x
_SAFE = [
    (_np_scalar, "numpy.core.multiarray.scalar"),
    np.dtype,
    (np.dtype, "numpy.dtype"),
    codecs.encode,
    (codecs.encode, "_codecs.encode"),
] + [type(np.dtype(t)) for t in ("float32", "float64", "int64", "int32")]


def load_state_dict(pth: Path) -> dict:
    with torch.serialization.safe_globals(_SAFE):
        ck = torch.load(pth, map_location="cpu", weights_only=True)
    return ck["model"] if isinstance(ck, dict) and "model" in ck else ck


def build_model(pth: Path, device: str = "cpu") -> torch.nn.Module:
    """torchvision Mask R-CNN (ResNet-50 FPN, 4 classes incl. background) with Davis weights."""
    model = maskrcnn_resnet50_fpn(weights=None, weights_backbone=None, num_classes=len(CLASSES) + 1,
                                  # ArcGIS fed raw 200 px chips; don't let torchvision upscale them to 800 px.
                                  min_size=200, max_size=200)
    sd = load_state_dict(pth)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"weights mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    return model.eval().to(device)


def emd(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def adapt_channels(model: torch.nn.Module, n: int) -> torch.nn.Module:
    """Widen the first conv to n input bands. Extra bands start from the mean of the
    pretrained RGB filters (scaled so the layer's output level is unchanged) and get
    neutral normalisation."""
    conv = model.backbone.body.conv1
    if conv.in_channels == n:
        return model
    new = torch.nn.Conv2d(n, conv.out_channels, conv.kernel_size, conv.stride, conv.padding, bias=False).to(conv.weight.device)
    with torch.no_grad():
        w = conv.weight
        new.weight[:, :3] = w * 3 / n
        new.weight[:, 3:] = w.mean(1, keepdim=True) * 3 / n
    model.backbone.body.conv1 = new
    model.transform.image_mean = list(model.transform.image_mean[:3]) + [0.45] * (n - 3)
    model.transform.image_std = list(model.transform.image_std[:3]) + [0.225] * (n - 3)
    return model
