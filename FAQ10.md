# PaddleDetection中Yolo系列修改anchor的方法
* 本文档介绍如何使用PaddleDetection的训练配置文件来实现YOLOv3/v4模型的anchor自定义设置
* Anchor修改主要分为两步，以YOLOv3为例

##### 1. Step 1: YOLOv3Head配置修改
```
YOLOv3Head:
  anchor_masks: [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
  anchors: [[10, 13], [16, 30], [33, 23],          # 修改此处
            [30, 61], [62, 45], [59, 119],         # 修改此处
            [116, 90], [156, 198], [373, 326]]     # 修改此处
  norm_decay: 0.
  yolo_loss: YOLOv3Loss
  nms:
    background_label: -1
    keep_top_k: 100
    nms_threshold: 0.45
    nms_top_k: 1000
    normalized: false
    score_threshold: 0.01
  drop_block: true
```
##### 2. Step 2: Reader.yml配置修改
需要额外注意的是，如果训练的yolov3.yml中使用了"use_fine_grained_loss: true"这个选项，则对应的TrainReader中会用到"Gt2YoloTarget"这一预处理函数
```
    # Gt2YoloTarget is only used when use_fine_grained_loss set as true,
    # this operator will be deleted automatically if use_fine_grained_loss
    # is set as false
    - !Gt2YoloTarget
      anchor_masks: [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
      anchors: [[10, 13], [16, 30], [33, 23],             # 修改此处，与上一步的设置需一致
                [30, 61], [62, 45], [59, 119],            # 修改此处，与上一步的设置需一致
                [116, 90], [156, 198], [373, 326]]        # 修改此处，与上一步的设置需一致
      downsample_ratios: [32, 16, 8]
```