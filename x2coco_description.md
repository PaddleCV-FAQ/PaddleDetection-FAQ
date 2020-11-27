Paddle Detection从入门到入土：

当我们需要告诉别人认识某样物品的时候，需要给对方一个实物来参照或提供一些照片、视频，让对方认识到该物品的特征。深度学习也是这样的，你需要告诉“电脑”这玩意儿长什么样，“电脑”才能学习、认识它。

为神经网络提供数据集就是告诉“电脑”这东西长什么样，训练神经网络就是让对方学习其特征。准备数据集是头等大事，那怎么做呢？

数据集就像菜，有别人做好的，也有自己做的。我们经常会见到COCO数据集和VOC数据集，本文先介绍VOC2007数据集的格式。

* VOC
    * JPEGImages
        * IMG_7369.JPG
        * IMG_7370.JPG
        * .....用于存放图片......
        * IMG_7371.JPG
        * IMG_7381.JPG
![图片](https://uploader.shimo.im/f/1BE35LG1JGDVVGNR.png!thumbnail)
    * Annatations
        * IMG_7369.XML
        * IMG_7370.XML
        * .....用于存放图片标注......
        * IMG_7371.XML
        * IMG_7381.XML
![图片](https://uploader.shimo.im/f/93cZVrePhbceiJys.png!thumbnail)
    * ImageSets
        * test.txt
        * train.txt
        * val.txt
        * trainval.txt
        * label_list.txt
![图片](https://uploader.shimo.im/f/164p7i5G4MemJCnW.png!thumbnail)


不同工具生成的ImageSets内容不同，不过基本是大同小异，其作用都是为了划分哪些图片用于训练、哪些用于测试。

工欲善其事，必先利其器。现一般使用labelImg、labelme、vott，国内有精灵标注等产品。都是可以生成VOC数据集的。PaddlePaddle文档中基本使用labelme。

如果你想使用代码划分训练集测试集，可以借鉴如下代码：

```python
# 根据个人情况更改root_path（根路径）、train_percent、trainval_percent （分割训练集、测试集的百分比）
import os
import random
import sys
from tqdm import tqdm
root_path = 'work/data/VOC'
xmlfilepath = root_path + '/Annotations'
txtsavepath = root_path 
if not os.path.exists(root_path):
    print("cannot find such directory: " + root_path)
    exit()
if not os.path.exists(txtsavepath):
    os.makedirs(txtsavepath)
trainval_percent = 0.9
train_percent = 0.8
total_xml = os.listdir(xmlfilepath)[:12000]
num = len(total_xml)
list = range(num)
tv = int(num * trainval_percent)
tr = int(tv * train_percent)
trainval = random.sample(list, tv)
train = random.sample(trainval, tr)
print("train and val size:", tv)
print("train size:", tr)
ftrainval = open(txtsavepath + '/trainval.txt', 'w')
ftest = open(txtsavepath + '/test.txt', 'w')
ftrain = open(txtsavepath + '/train_list.txt', 'w')
fval = open(txtsavepath + '/val_list.txt', 'w')
for i in tqdm(range(num)):
    name = total_xml[i][:-4] 
    if i in trainval:
        ftrainval.write(name)
        if i in train:
            ftrain.write('JPEGImages/' + name + '.jpg Annotations/' + name + '.xml' + '\n')
        else:
            fval.write('JPEGImages/' + name + '.jpg Annotations/' + name + '.xml' + '\n')
    else:
        ftest.write(name)
ftrainval.close()
ftrain.close()
fval.close()
ftest.close()
```
如果你已经熟悉了VOC数据集，并且知道VOC数据集是使用图片名.xml文件来描述图片中的标注信息，那么我们可以认识一下COCO是如何使用JSON描述标注信息的。

* COCO
    * Annotations
        * train.json
        * val.json
        * test.json
    * trainset
    * valset
    * testset

其中JSON文件长这个样：

```json
{
   "info": info,
   "images": [image],
   "annotations": [annotation],
   "licenses": [license],
}
info{
   "year": int,
   "version": str,
   "description": str,
   "contributor": str,
   "url": str,
   "date_created": datetime,
   }
image{
       "id": int,
       "width": int,
       "height": int,
       "file_name": str,
       "license": int,
       "flickr_url": str,
       "coco_url": str,
       "date_captured": datetime,
   }
license{
   "id": int,
   "name": str,
   "url": str,
   }
```
在JSON中

* info： 数据集信息
* license：你获得的图片的许可证
* images：每张图片的信息，包含文件名、宽、高、id
* categories：目标的类别信息

如果你打开VOC数据集中的XML文件，可以看到一个XML文件包含如下信息：

* folder: 文件夹
* filename：文件名
* path：路径
* source：来源
* size：图片大小
* segmented：图像分割会用到，本文仅以目标检测（bounding box为例进行介绍）
* object：一个xml文件可以有多个object，每个object表示一个box，每个box有如下信息组成：
    * name：改box框出来的object属于哪一类，例如Apple
    * bndbox：给出左上角和右下角的坐标
    * truncated、difficult
```xml
<annotation>
    <folder>文件夹目录</folder>
    <filename>图片名.jpg</filename>
    <path>path_to\at002eg001.jpg</path>
    <source>
        <database>Unknown</database>
    </source>
    <size>
        <width>550</width>
        <height>518</height>
        <depth>3</depth>
    </size>
    <segmented>0</segmented>
    <object>
        <name>Apple</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>292</xmin>
            <ymin>218</ymin>
            <xmax>410</xmax>
            <ymax>331</ymax>
        </bndbox>
    </object>
    <object>
        ...
    </object>
</annotation>
```

以前我们需要写代码手动将xml中的标注信息一一对应格式化写入json文件中。而PaddlePaddle为我们提供了多种数据集转化COCO的代码，直接调用X2COCO.py(在这里：)即可。

（1）labelme数据转换为COCO数据：根据自己需要更改train_proportion、val_proportion、test_proportion 来分割训练集、测试集大小

```shell
python tools/x2coco.py \
                --dataset_type labelme \
                --json_input_dir ./labelme_annos/ \
                --image_input_dir ./labelme_imgs/ \
                --output_dir ./cocome/ \
                --train_proportion 0.8 \
                --val_proportion 0.2 \
                --test_proportion 0.0
```
（2）VOC数据转换为COCO数据：
```shell
python tools/x2coco.py \
        --dataset_type voc \
        --voc_anno_dir path/to/VOCdevkit/VOC2007/Annotations/ \
        --voc_anno_list path/to/VOCdevkit/VOC2007/ImageSets/Main/trainval.txt \
        --voc_label_list dataset/voc/label_list.txt \
        --voc_out_name voc_train.json
```
最后，我本人使用的是VOTT，可以查看哪些图片未标注，同时可以导出多种格式，推荐大家使用。

![图片](https://uploader.shimo.im/f/qGoDaQjD39qZ4AMY.png!thumbnail)


