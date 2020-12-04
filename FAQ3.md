# 单类目标的召回和精度统计
* 本文档使用pycocotools计算单类别的precision和recall
* pycocotools会自动解析标签文件和模型预测所产生的json文件
* 本文档以标签文件'instances_val2017.json'和预测文件'retinanet_r50_val_results.bbox.json'为例

```python
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

gt_file = 'instances_val2017.json'
res_file = 'retinanet_r50_val_results'

cocoGt = COCO(gt_file)
cocoDt = cocoGt.loadRes(res_file)
cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()

# from https://github.com/facebookresearch/detectron2/
# precision: (iou, recall, cls, area range, max dets)
precisions = cocoEval.eval['precision']
# recall: (iou, cls, area range, max dets)
recalls = cocoEval.eval['recall']
assert len(cocoGt.getCatIds()) == precisions.shape[2]

print('\t'.join(['cls_name', 'mAP', 'ap50', 'ap75', 'ar50', 'ar75']))
for idx, catId in enumerate(cocoGt.getCatIds()):
    nm = cocoGt.loadCats(catId)[0]
    # AP calculate
    # area range index 0: all area ranges
    # max dets index -1: typically 100 per image
    # precision date shape: iouThrs, recThrs, catIds, areaRng, maxDets
    precision = precisions[:, :, idx, 0, -1]
    precision = precision[precision > -1]
    # mAP: AP at IoU=0.50:0.05:0.95
    if precision.size:
        mAP = np.mean(precision)
    else:
        mAP = float('nan')
    # ap50: AP at IoU=0.50
    ap50 = precision[0]
    # ap75: AP at IoU=0.75
    ap75 = precision[5]
    
    # AR calculate
    # recall data shape: iouThrs, catIds, areaRng, maxDets
    recall = recalls[:, idx, 0, -1]
    recall = recall[recall > -1]
    # ar50: AP at IoU=0.50
    ar50 = recall[0]
    # ar75: AP at IoU=0.75
    ap75 = recall[5]
    print('\t'.join([f'{nm["name"]}', 
                     f'{float(mAP):0.3f}',
                     f'{float(ap50):0.3f}',                 
                     f'{float(ap75):0.3f}', 
                     f'{float(ar50):0.3f}', 
                     f'{float(ap75):0.3f}']))
```
