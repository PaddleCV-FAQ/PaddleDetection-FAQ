在目标检测中, 常见的两个评估指标有AP或者AR, 下图是[coco官方的评估指标](./coco_evaluations.png)


#IoU(Intersection over Union)
Intersection over Union是一种测量在特定数据集中检测相应物体准确度的一个标准。我们可以在很多物体检测挑战中，例如PASCAL VOC challenge中看多很多使用该标准的做法。

IoU是一个简单的测量标准，只要是在输出中得出一个预测范围(bounding boxex)的任务都可以用IoU来进行测量。为了可以使IoU用于测量任意大小形状的物体检测，我们需要：
1、 ground-truth bounding boxes（人为在训练集图像中标出要检测物体的大概范围）；
2、我们的算法得出的结果范围。

计算的公式为 实际预测的 bounding boxes和预测的Bounding boxes的交集除于他们的并集。如果IoU > 阈值则预测为真确, 如果IoU < 阈值则预测为错误。
具体可以参考[下图](./IoU.png)

要理解mAP, 首先我们得了解准确率(Precision)和召回率(Recall).
下图是一个[混淆矩阵](./Confusion_Matrix)

Recall   召回率（查全率）。表示正确识别物体A的个数占测试集中物体A的总个数的百分数，即所有正例中预测正确的概率，Recall = TP / (TP+FN)

Precision 精确率（查准率）。表示正确识别物体A的个数占总识别出的物体个数n的百分数，即预测为正例中预测正确的概率，Precision = TP / (TP+FP)

fp :false positive误报，即预测错误

fn :false negative漏报，即没有预测到

- mAP(mean Average precision)
  - AP % AP at IoU=0.50:0.05:0.95 (primary challenge metric)
  - APIoU=.50 % AP at IoU=0.50 (PASCAL VOC metric)
  - APIoU=.75 % AP at IoU=0.75 (strict metric)
- AP Across Scales:
  - APsmall % AP for small objects: area < 322
  - APmedium % AP for medium objects: 322 < area < 962
  - APlarge % AP for large objects: area > 962
- Average Recall (AR):
  - ARmax=1 % AR given 1 detection per image
  - ARmax=10 % AR given 10 detections per image
  - ARmax=100 % AR given 100 detections per image
- AR Across Scales:
  - ARsmall % AR for small objects: area < 322
  - ARmedium % AR for medium objects: 322 < area < 962
  - ARlarge % AR for large objects: area > 962

1）除非另有说明，否则AP和AR在多个交汇点（IoU）值上取平均值。具体来说，我们使用10个IoU阈值0.50：0.05：0.95。这是对传统的一个突破，其中AP是在一个单一的0.50的IoU上计算的（这对应于我们的度量APIoU=.50 ）。超过均值的IoUs能让探测器更好定位（Averaging over IoUs rewards detectors with better localization.）。

2）AP是所有类别的平均值。传统上，这被称为“平均精确度”（mAP，mean average precision）。我们没有区分AP和mAP（同样是AR和mAR），并假定从上下文中可以清楚地看出差异。

3)AP（所有10个IoU阈值和所有80个类别的平均值）将决定赢家。在考虑COCO性能时，这应该被认为是最重要的一个指标。

4)在COCO中，比大物体相比有更多的小物体。具体地说，大约41％的物体很小（面积<322），34％是中等（322 < area < 962)），24％大（area > 962）。测量的面积（area）是分割掩码（segmentation mask）中的像素数量。

5）AR是在每个图像中检测到固定数量的最大召回（recall），在类别和IoU上平均。AR与提案评估（proposal evaluation）中使用的同名度量相关，但是按类别计算。

6）所有度量标准允许每个图像（在所有类别中）最多100个最高得分检测进行计算。

7）除了IoU计算（分别在框（box）或掩码（mask）上执行）之外，用边界框和分割掩码检测的评估度量在所有方面是相同的。

