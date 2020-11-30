# 通过自制数据集使用PaddleDetection训练
* 我们可以通过软件标注出各种数据集，下面以目标检测为例，使用以下代码可以在导出的voc格式的xml文件中剔除未标记的文件
* （当然你的文件里最好都是全部标准好的xml文件）

```python
import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import sys
import argparse

parse = argparse.ArgumentParser()
parse.add_argument('--xml_dir',type=str,default='Annotations')   #这个文件夹下放置的是你导出的xml文件
args = parse.parse_args()

def pop_empty():
    assert args.xml_dir is not None , 'xml file can,t be none'
    xml_file_list = os.listdir(args.xml_dir)
    for xml in xml_file_list:
        xml_file = open('%s/%s'%(args.xml_dir,xml))
        tree=ET.parse(xml_file)
        root = tree.getroot()
        size = root.find('size')
        if size is None:
            os.remove(xml_file.name)

if __name__ == '__main__':
    args.xml_dir = 'outputs'
    pop_empty()
```
* 我们的数据集文件夹是这样的

```
.
├── Annotations
│   ├── test
|   |   ├── 0005.xml
|   |   └── 0006.xml
│   ├── train
|   |   ├── 0004.xml
│   │   └── 0003.xml
│   └── val
│       ├── 0001.xml
│       └── 0002.xml
├── ImageSets
│   ├── label_list.txt
│   ├── test.txt
│   ├── train.txt
│   └── val.txt
└── JPEGImages
    ├── test
    |   ├── 0005.jpg
    |   └── 0006.jpg
    ├── train
    |   ├── 0004.jpg
    │   └── 0003.jpg
    └── val
        ├── 0001.jpg
        └── 0002.jpg
```

* Annotations文件夹里面放的是xml文件
* JPEGImages文件夹里放的是对应于xml的图片文件
* ImageSet 内是四个文件，其中test.txt , train.txt , val.txt 可以通过以下代码生成，label_list.txt文件内是记录类别的
```python
#如果是自己标注的数据集用这个则可以生成，生成train，test和eval  txt文件
import os
import argparse

parse = argparse.ArgumentParser()
parse.add_argument('--xml_file_path',type=str,default='Annotations')
parse.add_argument('--img_file_path',type=str,default='JPEGImages')
parse.add_argument('--output_file',type=str,default='ImageSets')
args = parse.parse_args()

def create_ImageSets():
    Annotations_file_list = os.listdir(args.xml_file_path)
    JPEFImages_file_list  = os.listdir(args.img_file_path) 
    if not os.path.exists(args.output_file):
        os.makedirs(args.output_file)
    for j in range(len(Annotations_file_list)):
        xmlfile_path = args.xml_file_path +'/'+Annotations_file_list[j]
        imgfile_path = args.img_file_path +'/'+JPEFImages_file_list[j]
        imgfiles = os.listdir(xmlfile_path)
        xmlfiles = os.listdir(imgfile_path)
        if(len(imgfiles) == len(xmlfiles)):
            lines = []   
            for i in range(len(imgfiles)):
                line = '../JPEGImages/'+ JPEFImages_file_list[j] + '/' + imgfiles[i]  + ' '+ '../Annotations/' + Annotations_file_list[j] + '/' + os.path.splitext(imgfiles[i])[0] +'.xml'                    # 循环读取路径下的文件并筛选输出
                lines.append(line)
        else:
            print('Annotations文件夹下文件需要和JPEFImages文件夹下文件对应')
        with open(args.output_file+'/'+Annotations_file_list[j]+'.txt','w') as f:
            for i in range(len(lines)):
                f.write(lines[i])
                f.write('\n')


if __name__ == '__main__':
    create_ImageSets()
```

