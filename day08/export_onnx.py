#!/usr/bin/env python3
"""Day 8: export two ONNX graphs from a random-weight resnet18.

model_only.onnx            input "tensor" float32 NCHW [1,3,480,640] -> "logits" [1,1000]
model_with_preprocess.onnx input "frame"  uint8   NHWC [1,720,1280,3] -> "logits" [1,1000]

weights=None: this is a latency test, not an accuracy test. The random-weight
network runs the exact same FLOPs and layer graph as a trained resnet18, so
build/inference timings are representative; the logits themselves are
meaningless (see e2e.py's correctness check, which compares A vs B logits,
not logits vs ground truth).
"""
import inspect

import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class PreprocessAndModel(nn.Module):
    """Folds camera preprocessing into the graph so the engine owns it."""

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1))

    def forward(self, frame: torch.Tensor) -> torch.Tensor:
        # frame: uint8 [1,720,1280,3] NHWC, RGB
        x = frame.float()
        x = x.permute(0, 3, 1, 2)  # NHWC -> NCHW
        x = F.interpolate(
            x, size=(480, 640), mode="bilinear", align_corners=False, antialias=False
        )
        x = x / 255.0
        x = (x - self.mean) / self.std
        return self.backbone(x)


def export_kwargs(input_names, output_names):
    kwargs = dict(
        input_names=input_names,
        output_names=output_names,
        opset_version=17,
        do_constant_folding=True,
    )
    # torch >= 2.5 exposes a `dynamo` switch on torch.onnx.export. Force the
    # legacy TorchScript-based exporter for a plain, predictable graph (no
    # dynamic_axes anywhere -- both graphs use fixed shapes). Older torch has
    # no such kwarg at all, hence the signature probe instead of a version
    # string compare.
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        kwargs["dynamo"] = False
    return kwargs


def describe(path: str) -> None:
    model = onnx.load(path)
    onnx.checker.check_model(model)
    print(f"== {path} (checker OK) ==")
    for tag, tensors in (("input", model.graph.input), ("output", model.graph.output)):
        for t in tensors:
            dtype = onnx.helper.tensor_dtype_to_np_dtype(t.type.tensor_type.elem_type)
            dims = [d.dim_value for d in t.type.tensor_type.shape.dim]
            print(f"  {tag:6s} {t.name!r:10s} dtype={dtype} shape={dims}")

    # Measure the resize semantics instead of just asserting them in
    # contract.md: the exporter's choice of coordinate_transformation_mode
    # for align_corners=False with an explicit output `size` is
    # "pytorch_half_pixel", not plain "half_pixel" -- they differ whenever an
    # output spatial dim is 1 (not our case here, but worth reading off the
    # actual graph rather than assuming).
    for node in model.graph.node:
        if node.op_type == "Resize":
            attrs = {}
            for a in node.attribute:
                v = onnx.helper.get_attribute_value(a)
                attrs[a.name] = v.decode() if isinstance(v, bytes) else v
            print(f"  Resize node {node.name!r}: attrs={attrs}")


def main() -> None:
    torch.manual_seed(0)
    backbone = torchvision.models.resnet18(weights=None)
    backbone.eval()

    # (a) model_only: input is already a preprocessed float tensor.
    tensor_in = torch.zeros(1, 3, 480, 640, dtype=torch.float32)
    torch.onnx.export(
        backbone,
        (tensor_in,),
        "model_only.onnx",
        **export_kwargs(["tensor"], ["logits"]),
    )

    # (b) model_with_preprocess: input is the raw camera frame.
    wrapped = PreprocessAndModel(backbone)
    wrapped.eval()
    frame_in = torch.zeros(1, 720, 1280, 3, dtype=torch.uint8)
    torch.onnx.export(
        wrapped,
        (frame_in,),
        "model_with_preprocess.onnx",
        **export_kwargs(["frame"], ["logits"]),
    )

    describe("model_only.onnx")
    describe("model_with_preprocess.onnx")


if __name__ == "__main__":
    main()
