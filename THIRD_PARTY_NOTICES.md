# Third-party notices

## PyTorchVideo

The `slowfast_r50` model definition from Facebook Research's PyTorchVideo was
ported and restructured into the local PyTorch implementation in
`src/yaslowfast/models.py`. This includes the architecture, module naming,
pooling, and initialization. The implementation has no dependency on
PyTorchVideo, torchvision, or fvcore for installation, execution, development,
tests, or export.

- Repository: https://github.com/facebookresearch/pytorchvideo
- Reference commit: `2ac6ed361309cb25144636ad82dd8bd6a4970e0c`
- Referenced files: `pytorchvideo/models/{slowfast,resnet,stem,head,net,weight_init}.py`
- Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
- License: Apache License 2.0, reproduced in `LICENSE`.

Changes: ported and restructured the SlowFast-R50 model definition into a single
pure-PyTorch module; fixed the classification head to two outputs for binary
classification; removed PyTorchVideo, torchvision, and fvcore dependencies from
runtime and verification tooling; exposed two tensor inputs; added input
validation, optional adaptive pooling, and explicit model configuration. The
original parameter names are retained for compatibility with the original
architecture. Weights are randomly initialized for training from scratch. No
pretrained weights are included or downloaded. The upstream reference has no
root NOTICE file.

The original copyright notice, Apache License 2.0 text, and record of changes are
retained to document the origin and applicable license of the ported code. These
notices and the license text are included in the distribution. They do not add a
package dependency. No implementation or verification tool installs or imports
PyTorchVideo, torchvision, or fvcore.
