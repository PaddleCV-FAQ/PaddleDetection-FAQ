# Kmeans聚类函数

为什么要进行anchor boxes聚类?

首先，这么做可以让我们对数据集中anchor boxes的情况有比较直观的了解，假设anchor boxes聚类结果都比较小，网络选择就要慎重考虑小目标问题了。

其次，原本的配置文件中，anchor boxes的选择是根据coco数据集计算得出的，因此在很多场景下（请注意，不是全部！）将聚类后的结果对默认设置进行替换，模型性能会有比较大的提升。

因此，这里整理了VOC和COCO格式数据集上，anchor boxes的聚类函数，只需替换最后的目录为需要进行计算的对应格式数据集目录即可，从而方便读者在Jupyter Notebook中使用，如需使用脚本，请自行加入main函数。

## VOC格式

参考项目：[PaddleX助力无人驾驶（基于YOLOv3的车辆检测和车道线分割） - Baidu AI Studio - 人工智能学习与实训社区](https://aistudio.baidu.com/aistudio/projectdetail/464339)

```python
## 使用ET模块解析xml标注文件

import xml.etree.ElementTree as ET
import numpy as np

tar_size = 608

def load_one_info(name):

    filename = os.path.join(base, 'Annotations', name)

    tree = ET.parse(filename)
    size = tree.find('size')
    width = float(size.find('width').text)
    height = float(size.find('height').text)
    ratio = min(tar_size / width, tar_size / height)

    Objects = tree.findall('object')
    objs_num = len(Objects)
    Boxes = np.zeros((objs_num, 4), dtype=np.float32)
    True_classes = np.zeros((objs_num), dtype=np.float32)
    
    result = []
    for i, obj in enumerate(Objects):

        bbox = obj.find('bndbox')

        x_min = float(bbox.find('xmin').text) - 1
        y_min = float(bbox.find('ymin').text) - 1
        x_max = float(bbox.find('xmax').text) - 1
        y_max = float(bbox.find('ymax').text) - 1

        w = ratio * (x_max - x_min)
        h = ratio * (y_max - y_min)
        result.append([w, h])

    return result

def iou(box, clusters):

    x = np.minimum(clusters[:, 0], box[0])
    y = np.minimum(clusters[:, 1], box[1])
    if np.count_nonzero(x == 0) > 0 or np.count_nonzero(y == 0) > 0:
        return 0
        raise ValueError("Box has no area")

    intersection = x * y
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]

    iou_ = np.true_divide(intersection, box_area +
                          cluster_area - intersection + 1e-10)
    # iou_ = intersection / (box_area + cluster_area - intersection + 1e-10)

    return iou_


def avg_iou(boxes, clusters):

    return np.mean([np.max(iou(boxes[i], clusters)) for i in range(boxes.shape[0])])


def translate_boxes(boxes):

    new_boxes = boxes.copy()
    for row in range(new_boxes.shape[0]):
        new_boxes[row][2] = np.abs(new_boxes[row][2] - new_boxes[row][0])
        new_boxes[row][3] = np.abs(new_boxes[row][3] - new_boxes[row][1])
    return np.delete(new_boxes, [0, 1], axis=1)


def kmeans(boxes, k, dist=np.median):

    rows = boxes.shape[0]

    distances = np.empty((rows, k))
    last_clusters = np.zeros((rows,))

    np.random.seed()

    # the Forgy method will fail if the whole array contains the same rows
    clusters = boxes[np.random.choice(rows, k, replace=False)]

    while True:
        for row in range(rows):
            distances[row] = 1 - iou(boxes[row], clusters)

        nearest_clusters = np.argmin(distances, axis=1)

        if (last_clusters == nearest_clusters).all():
            break

        for cluster in range(k):
            clusters[cluster] = dist(
                boxes[nearest_clusters == cluster], axis=0)

        last_clusters = nearest_clusters

    return clusters

def get_kmeans(anno, cluster_num=9):

    anchors = kmeans(anno, cluster_num)
    ave_iou = avg_iou(anno, anchors)

    anchors = anchors.astype('int').tolist()

    anchors = sorted(anchors, key=lambda x: x[0] * x[1])

    return anchors, ave_iou
```

```python
result = []
# 这里要将VOC格式数据集的目录指定为base目录
for name in os.listdir(os.path.join(base, 'Annotations')):  
    result.extend(load_one_info(name))

result = np.array(result)
anchors, ave_iou = get_kmeans(result, 9)

anchor_string = ''
anchor_sizes = []
for anchor in anchors:
    anchor_string += '{},{}, '.format(anchor[0], anchor[1])
    anchor_sizes.append([anchor[0], anchor[1]])
anchor_string = anchor_string[:-2]

print('anchors are:')
print(anchor_string)
print('the average iou is:')
print(ave_iou)
```



## COCO格式

参考项目：[CascadeRCNN和YOLOv3_Enhance的布匹瑕疵检测模型训练部署 - Baidu AI Studio - 人工智能学习与实训社区](https://aistudio.baidu.com/aistudio/projectdetail/532715)

```python
def iou(box, clusters):
    """
    Calculates the Intersection over Union (IoU) between a box and k clusters.
    :param box: tuple or array, shifted to the origin (i. e. width and height)
    :param clusters: numpy array of shape (k, 2) where k is the number of clusters
    :return: numpy array of shape (k, 0) where k is the number of clusters
    """
    x = np.minimum(clusters[:, 0], box[0])
    y = np.minimum(clusters[:, 1], box[1])
    if np.count_nonzero(x == 0) > 0 or np.count_nonzero(y == 0) > 0:
        raise ValueError("Box has no area")
 
    intersection = x * y
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]
 
    iou_ = intersection / (box_area + cluster_area - intersection)
 
    return iou_
 
 
def avg_iou(boxes, clusters):
    """
    Calculates the average Intersection over Union (IoU) between a numpy array of boxes and k clusters.
    :param boxes: numpy array of shape (r, 2), where r is the number of rows
    :param clusters: numpy array of shape (k, 2) where k is the number of clusters
    :return: average IoU as a single float
    """
    return np.mean([np.max(iou(boxes[i], clusters)) for i in range(boxes.shape[0])])
 
 
def translate_boxes(boxes):
    """
    Translates all the boxes to the origin.
    :param boxes: numpy array of shape (r, 4)
    :return: numpy array of shape (r, 2)
    """
    new_boxes = boxes.copy()
    for row in range(new_boxes.shape[0]):
        new_boxes[row][2] = np.abs(new_boxes[row][2] - new_boxes[row][0])
        new_boxes[row][3] = np.abs(new_boxes[row][3] - new_boxes[row][1])
    return np.delete(new_boxes, [0, 1], axis=1)
 
 
def kmeans(boxes, k, dist=np.median):
    """
    Calculates k-means clustering with the Intersection over Union (IoU) metric.
    :param boxes: numpy array of shape (r, 2), where r is the number of rows
    :param k: number of clusters
    :param dist: distance function
    :return: numpy array of shape (k, 2)
    """
    rows = boxes.shape[0]
 
    distances = np.empty((rows, k))
    last_clusters = np.zeros((rows,))
 
    np.random.seed()
 
    # the Forgy method will fail if the whole array contains the same rows
    clusters = boxes[np.random.choice(rows, k, replace=False)]
 
    while True:
        for row in range(rows):
            distances[row] = 1 - iou(boxes[row], clusters)
 
        nearest_clusters = np.argmin(distances, axis=1)
 
        if (last_clusters == nearest_clusters).all():
            break
 
        for cluster in range(k):
            clusters[cluster] = dist(boxes[nearest_clusters == cluster], axis=0)
 
        last_clusters = nearest_clusters
 
    return clusters
```

```python
CLUSTERS = 9

def load_dataset(path):
    dataset = []
    with open(os.path.join(path, 'annotations',  'instances_train2017.json')) as f:
        json_file = json.load(f) 
        for item in json_file['images']:            
            height = item['height']
            width = item['width']
            for objs in json_file['annotations']:
                if(objs['image_id'] == item['id']):
                    bbox = objs['bbox']
                    xmin = int(bbox[0]) / width
                    ymin = int(bbox[1]) / height
                    xmax = int(bbox[0] + bbox[2]) / width
                    ymax = int(bbox[1] + bbox[3]) / height
 
                    xmin = np.float64(xmin)
                    ymin = np.float64(ymin)
                    xmax = np.float64(xmax)
                    ymax = np.float64(ymax)
                    if xmax == xmin or ymax == ymin:
                        print(item['file_name'])
                    dataset.append([xmax - xmin, ymax - ymin])
        f.close()
    return np.array(dataset)
 

#print(__file__)
# 这里指定的就是coco数据集所在目录
data = load_dataset('PaddleDetection/dataset/coco')

out = kmeans(data, k=CLUSTERS)*608
print(out)
print("Accuracy: {:.2f}%".format(avg_iou(data, out/608) * 100))
print("Boxes:\n {}-{}".format(out[:, 0], out[:, 1]))
 
ratios = np.around(out[:, 0] / out[:, 1], decimals=2).tolist()
print("Ratios:\n {}".format(sorted(ratios)))
```

