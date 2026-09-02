# Day 8 -- input contract

Both engines return `"logits"` float32 `[1, 1000]` (meaningless values -- the
backbone has random weights; `e2e.py` compares A's logits against B's, not
against ground truth). The difference is what crosses the ONNX/TensorRT
boundary versus what stays in plain PyTorch/CUDA code outside the engine.

## model_only.plan

| field         | value                                                    |
|---------------|-----------------------------------------------------------|
| input name    | `tensor`                                                   |
| layout        | NCHW                                                       |
| dtype         | float32                                                    |
| shape         | [1, 3, 480, 640]                                           |
| color order   | RGB                                                        |
| value range   | normalized: `(x/255 - mean) / std`                          |
| mean          | (0.485, 0.456, 0.406)                                       |
| std           | (0.229, 0.224, 0.225)                                       |
| resize        | done OUTSIDE the engine (caller's job)                       |
| inside engine | resnet18 only                                               |

## model_with_preprocess.plan

| field         | value                                                                        |
|---------------|--------------------------------------------------------------------------------|
| input name    | `frame`                                                                        |
| layout        | NHWC                                                                           |
| dtype         | uint8                                                                          |
| shape         | [1, 720, 1280, 3]                                                              |
| color order   | RGB                                                                            |
| value range   | raw camera bytes, 0-255                                                        |
| resize        | bilinear, `align_corners=False`, `antialias=False`, (720,1280) -> (480,640), INSIDE the graph. Exact `coordinate_transformation_mode` (e.g. `pytorch_half_pixel` vs plain `half_pixel` -- these differ when an output spatial dim is 1) is NOT asserted here; `export_onnx.py`'s `describe()` prints the actual `Resize` node attributes from the exported graph -- read those instead of trusting this table. |
| normalize     | INSIDE the graph: cast to float, /255, subtract mean, divide std               |
| mean          | (0.485, 0.456, 0.406)                                                          |
| std           | (0.229, 0.224, 0.225)                                                          |
| inside engine | permute + resize + normalize + resnet18                                        |

## what lives inside the engine

| stage                       | model_only | model_with_preprocess |
|------------------------------|:----------:|:----------------------:|
| uint8 -> float cast          | outside    | inside                 |
| NHWC -> NCHW permute         | outside    | inside                 |
| resize 720x1280 -> 480x640   | outside    | inside                 |
| /255 normalize               | outside    | inside                 |
| mean/std normalize           | outside    | inside                 |
| resnet18 backbone            | inside     | inside                 |

If preprocessing runs outside the engine, `t_e2e` includes a Python/PyTorch
round trip that TensorRT never sees and never optimizes or fuses into the
graph. That gap is what this day's numbers (`e2e.py`, `outside_*` vs
`inside_gpu_frame`) are meant to make visible.
