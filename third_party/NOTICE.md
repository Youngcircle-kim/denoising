# Third-party code and checkpoint provenance

`mstmn_train/flow.py` adapts the six-stage SPyNet architecture and checkpoint
conventions in [XPixelGroup/BasicSR](https://github.com/XPixelGroup/BasicSR/blob/master/basicsr/archs/spynet_arch.py).
BasicSR is distributed under Apache-2.0; its license is retained in
[BasicSR-LICENSE.txt](BasicSR-LICENSE.txt). Changes include standalone warping,
explicit safe local loading, frozen flow inference, and the RAW/RGB bridge.

The Sintel flow checkpoint is served by the
[VRT project](https://github.com/JingyunLiang/VRT/releases/download/v0.0/spynet_sintel_final-3d2a1287.pth).
Its published filename SHA-256 prefix and observed full SHA-256 are recorded in
`research/SPyNet_checkpoint_receipt.json`. This is not an MSTMN checkpoint.
The actual checkpoint is excluded from Git.

The original SPyNet paper's [author repository](https://github.com/anuragranj/spynet)
uses Torch/Lua. Our selected PyTorch compatibility path is a reproduction choice;
the exact MSTMN flow implementation and pretrained checkpoint remain unknown.
