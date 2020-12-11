# 序言
<font size=4>在树莓派上目标检测模型只能部署SSD?能不能换成效果更好的YoloV3?那当然是——可以！</font>

<font size=4>只是[Paddle-Lite-Demo项目](https://gitee.com/paddlepaddle/Paddle-Lite-Demo)上仅提供了SSD模型的部署方法，换成YoloV3恐怕少不了一番周折……</font>

<font size=4>本项目提供了完整的安全帽检测YoloV3模型在树莓派上的部署教程，旨在帮助读者少走弯路，都来试试吧?</font>![file](https://ai-studio-static-online.cdn.bcebos.com/56bde642c52243bf973fb40142d6f1d7f39dc085146a4efbafe0e83f0a9ed057)

## 本文相关前置项目汇总
- [安全帽佩戴检测模型训练与一键部署（PaddleX、HubServing）](https://aistudio.baidu.com/aistudio/projectdetail/742090)
- [PaddleX、PP-Yolo：手把手教你训练、加密、部署目标检测模型](https://aistudio.baidu.com/aistudio/projectdetail/920753)
- [PaddleLite树莓派从0到1：安全帽检测小车部署（一）](https://aistudio.baidu.com/aistudio/projectdetail/1059610)
- [巡检告警机器人上线！PaddleLite安全帽检测小车部署（二）](https://aistudio.baidu.com/aistudio/projectdetail/1209733)
- [YoloV3检测模型在树莓派上的部署（PaddleX、PaddleLite）](https://aistudio.baidu.com/aistudio/projectdetail/1227445)

> - 注1：关于PaddleDetection如何进行迁移学习训练，并导出可在树莓派上部署的Lite模型，请查看前置项目[PaddleLite树莓派从0到1：安全帽检测小车部署（一）](https://aistudio.baidu.com/aistudio/projectdetail/1059610)
> - 注2：关于如何在设备上自行编译PaddleLite，请查看前置项目[YoloV3检测模型在树莓派上的部署（PaddleX、PaddleLite）](https://aistudio.baidu.com/aistudio/projectdetail/1227445)
> - 注3：本项目使用的是Raspberry Pi OS 64位操作系统，如与您使用的系统不同，请到[https://github.com/PaddlePaddle/Paddle-Lite/releases](https://github.com/PaddlePaddle/Paddle-Lite/releases)上选择或自行编译对应的预测库，但是部署代码无需调整。

# 模型准备
在项目挂载的数据集中，提供了基于`yolov3_mobilenetv3`训练的安全帽检测部署模型`det_yolov3_mobilenetv3.nb`


```python
!ls data/data62679/det_yolov3_mobilenetv3.nb
```

    data/data62679/det_yolov3_mobilenetv3.nb


# 环境准备
## 在树莓派上拉取Paddle-Lite V2.6.3代码
```bash
git clone https://gitee.com/paddlepaddle/paddle-lite
git checkout v2.6.3
```
## 直接解压或编译预测库
如果您的树莓派也是64位系统，可下载项目挂载数据集里的预编译库；如果不是，请参考[源码编译方法](https://paddle-lite.readthedocs.io/zh/latest/quick_start/release_lib.html#id2)自行编译


```python
# Paddle-Lite(ArmLinuxV8)FULL预编译库.zip
!ls data/data62679/Paddle-Lite*
```

    data/data62679/Paddle-Lite(ArmLinuxV8)FULL预编译库.zip


解压后paddle-lite目录结构如下：
```bash
pi@raspberrypi:~ $ ls paddle-lite/
build.lite.armlinux.armv8.gcc  cmake  CMakeLists.txt  docs  LICENSE  lite  metal  mobile  README_en.md  README.md  third-party  third-party-05b862.tar.gz  tools  web
```

## 解读`yolov3_detection.cc`
其实，在Paddle-Lite项目中已经给出了yolov3部署的demo，可参考[https://gitee.com/paddlepaddle/paddle-lite/blob/develop/lite/demo/cxx/yolov3_detection/yolov3_detection.cc](http://https://gitee.com/paddlepaddle/paddle-lite/blob/develop/lite/demo/cxx/yolov3_detection/yolov3_detection.cc)
但是`yolov3_detection.cc`只给出了一个文件，缺少相关的文档介绍，这里试着解读下代码的重点。

### 1. 先看`main()`函数
这也是近来几次课程里老师传授的重点了，先看`main()`函数有什么功能，可以很快理解代码的意图。
```c++
int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "[ERROR] usage: " << argv[0] << " model_file image_path\n";
    exit(1);
  }
  // 传入Lite模型文件
  std::string model_file = argv[1];
  // 对图片预测，传入图片文件
  std::string img_path = argv[2];
  // 执行预测
  RunModel(model_file, img_path);
  return 0;
}
所以，`main()`函数其实很简单，我们需要传入两个参数，模型文件路径和图片路径，然后执行预测。
```
### 2. 再看`RunModel()`函数
毫无疑问`RunModel(model_file, img_path)`的实现是这个文件的重点，这里增加了一些注释，<font color="red">重点要注意的是处理图片大小的两个内置类型变量</font>
```c++
void RunModel(std::string model_file, std::string img_path) {
  // 1. Set MobileConfig 读取模型文件
  MobileConfig config;
  config.set_model_from_file(model_file);

  // 2. Create PaddlePredictor by MobileConfig 创建Predictor
  std::shared_ptr<PaddlePredictor> predictor =
      CreatePaddlePredictor<MobileConfig>(config);
  // 设置模型输入，注意到608*608在树莓派3B上会耗尽资源，这里要进行调整
  const int in_width = 608;
  const int in_height = 608;

  // 3. Prepare input data from image
  // input 0
  std::unique_ptr<Tensor> input_tensor0(std::move(predictor->GetInput(0)));
  input_tensor0->Resize({1, 3, in_height, in_width});
  auto* data0 = input_tensor0->mutable_data<float>();
  // 使用OpenCV把图片读取出来
  cv::Mat img = imread(img_path, cv::IMREAD_COLOR);
  // 前处理部分，注意到传入了in_width和in_height
  pre_process(img, in_width, in_height, data0);
  // input1
  std::unique_ptr<Tensor> input_tensor1(std::move(predictor->GetInput(1)));
  input_tensor1->Resize({1, 2});
  auto* data1 = input_tensor1->mutable_data<int>();
  data1[0] = img.rows;
  data1[1] = img.cols;

  // 4. Run predictor 执行预测
  predictor->Run();

  // 5. Get output and post process 得到预测结果并进行后处理
  std::unique_ptr<const Tensor> output_tensor(
      std::move(predictor->GetOutput(0)));
  auto* outptr = output_tensor->data<float>();
  auto shape_out = output_tensor->shape();
  int64_t cnt = 1;
  for (auto& i : shape_out) {
    cnt *= i;
  }
  // 后处理部分，将检测结果和原图画框
  auto rec_out = detect_object(outptr, static_cast<int>(cnt / 6), 0.5f, img);
  // 保存画框后的图片
  std::string result_name =
      img_path.substr(0, img_path.find(".")) + "_yolov3_detection_result.jpg";
  cv::imwrite(result_name, img);
}
```
### 3. 前处理函数`pre_process()`
前处理函数的逻辑：得到一张图片，<font color="red">resize一番！</font>再做一些归一化处理，都是常规动作，所以最后的`neon_mean_scale()`这里也不赘述了。
```c++
void pre_process(const cv::Mat& img, int width, int height, float* data) {
  cv::Mat rgb_img;
  cv::cvtColor(img, rgb_img, cv::COLOR_BGR2RGB);
  cv::resize(
      rgb_img, rgb_img, cv::Size(width, height), 0.f, 0.f, cv::INTER_CUBIC);
  cv::Mat imgf;
  rgb_img.convertTo(imgf, CV_32FC3, 1 / 255.f);
  std::vector<float> mean = {0.485f, 0.456f, 0.406f};
  std::vector<float> scale = {0.229f, 0.224f, 0.225f};
  const float* dimg = reinterpret_cast<const float*>(imgf.data);
  neon_mean_scale(dimg, data, width * height, mean, scale);
}
```
### 4. 后处理函数`detect_object()`
后处理函数内容较多，但我们可以思考下，如果对后处理函数要进行改动，会改哪里?一般就是检测到目标，报个警之类的吧?那这时候，只要重点关注能在哪里加入（或直接利用）`if`语句吧?

至于截图?`RunModel()`里面单独拎出来了不是，紧接在后处理函数后面。
```c++
std::vector<Object> detect_object(const float* data,
                                  int count,
                                  float thresh,
                                  cv::Mat& image) {  // NOLINT
  if (data == nullptr) {
    std::cerr << "[ERROR] data can not be nullptr\n";
    exit(1);
  }
  std::vector<Object> rect_out;
  for (int iw = 0; iw < count; iw++) {
    int oriw = image.cols;
    int orih = image.rows;
    if (data[1] > thresh) {
      Object obj;
      int x = static_cast<int>(data[2]);
      int y = static_cast<int>(data[3]);
      int w = static_cast<int>(data[4] - data[2] + 1);
      int h = static_cast<int>(data[5] - data[3] + 1);
      cv::Rect rec_clip =
          cv::Rect(x, y, w, h) & cv::Rect(0, 0, image.cols, image.rows);
      obj.class_id = static_cast<int>(data[0]);
      obj.prob = data[1];
      obj.rec = rec_clip;
      // 下面就是个很理想的，可以考虑直接加入报警函数的地方
      if (w > 0 && h > 0 && obj.prob <= 1) {
        rect_out.push_back(obj);
        cv::rectangle(image, rec_clip, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
        std::string str_prob = std::to_string(obj.prob);
        std::string text = std::string(class_names[obj.class_id]) + ": " +
                           str_prob.substr(0, str_prob.find(".") + 4);
        int font_face = cv::FONT_HERSHEY_COMPLEX_SMALL;
        double font_scale = 1.f;
        int thickness = 1;
        cv::Size text_size =
            cv::getTextSize(text, font_face, font_scale, thickness, nullptr);
        float new_font_scale = w * 0.5 * font_scale / text_size.width;
        text_size = cv::getTextSize(
            text, font_face, new_font_scale, thickness, nullptr);
        cv::Point origin;
        origin.x = x + 3;
        origin.y = y + text_size.height + 3;
        cv::putText(image,
                    text,
                    origin,
                    font_face,
                    new_font_scale,
                    cv::Scalar(0, 255, 255),
                    thickness,
                    cv::LINE_AA);

        std::cout << "detection, image size: " << image.cols << ", "
                  << image.rows
                  << ", detect object: " << class_names[obj.class_id]
                  << ", score: " << obj.prob << ", location: x=" << x
                  << ", y=" << y << ", width=" << w << ", height=" << h
                  << std::endl;
      }
    }
    data += 6;
  }
  return rect_out;
}
```

## 改造`yolov3_detection.cc`
理解了`yolov3_detection.cc`的结构，梳理下改动的思路

<font size=4>1. 修改resize后图片的大小（必须的，否则树莓派3B资源就爆了）</font>

<font size=4>2. 写一个报警函数`warn()`——关于报警函数的写法和树莓派引脚接线，请查看[巡检告警机器人上线！PaddleLite安全帽检测小车部署（二）](https://aistudio.baidu.com/aistudio/projectdetail/1209733)</font>

<font size=4>3. 利用后处理的if判断语句，增加检测到未佩戴安全帽时才触发报警</font>

<font size=4>4. 截图存证，引入时间戳，这样可以连续保存函数</font>

<font size=4>5. 将输入图片的预测转换为直接输入视频流</font>

<font size=4>6. 使用`cmake`编译——关于`CMakeLists.txt`和`build.sh`的写法，请查看[巡检告警机器人上线！PaddleLite安全帽检测小车部署（二）](https://aistudio.baidu.com/aistudio/projectdetail/1209733)</font>

## 修改后的`yolov3_detection.cc`
篇幅所限，`yolov3_detection.cc`的修改就不赘述了，读者可以用前面的方法试着理解下。
```c++
// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// 注意这里增加了wiringPi库和计算时间的sys/time库

#include <iostream>
#include <vector>
#include <sys/time.h>
#include <stdlib.h>
#include <string.h>
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <wiringPi.h>
#include "paddle_api.h"  // NOLINT

#define	LED	17

using namespace paddle::lite_api;  // NOLINT

struct Object {
    cv::Rect rec;
    int class_id;
    float prob;
};

int64_t ShapeProduction(const shape_t& shape) {
    int64_t res = 1;
    for (auto i : shape) res *= i;
    return res;
}

inline int64_t get_current_us() {
    struct timeval time;
    gettimeofday(&time, NULL);
    return 1000000LL * (int64_t)time.tv_sec + (int64_t)time.tv_usec;
}

// 预测的标签是写死的，这里改成与训练数据集对应的标签
const char* class_names[] = { "head",        "helmet",      "person"
};

// 增加报警函数，注意使用的是BCM编码
void warn(void)
{
    wiringPiSetupSys();

    pinMode(LED, OUTPUT);

    for (int i = 1; i <= 6; i++)
    {
        digitalWrite(LED, HIGH);  // 启用
        delay(500); // 毫秒
        digitalWrite(LED, LOW);	  // 关
        delay(500);
    }
}

// fill tensor with mean and scale and trans layout: nhwc -> nchw, neon speed up
void neon_mean_scale(const float* din,
    float* dout,
    int size,
    const std::vector<float> mean,
    const std::vector<float> scale) {
    if (mean.size() != 3 || scale.size() != 3) {
        std::cerr << "[ERROR] mean or scale size must equal to 3\n";
        exit(1);
    }
    float32x4_t vmean0 = vdupq_n_f32(mean[0]);
    float32x4_t vmean1 = vdupq_n_f32(mean[1]);
    float32x4_t vmean2 = vdupq_n_f32(mean[2]);
    float32x4_t vscale0 = vdupq_n_f32(1.f / scale[0]);
    float32x4_t vscale1 = vdupq_n_f32(1.f / scale[1]);
    float32x4_t vscale2 = vdupq_n_f32(1.f / scale[2]);

    float* dout_c0 = dout;
    float* dout_c1 = dout + size;
    float* dout_c2 = dout + size * 2;

    int i = 0;
    for (; i < size - 3; i += 4) {
        float32x4x3_t vin3 = vld3q_f32(din);
        float32x4_t vsub0 = vsubq_f32(vin3.val[0], vmean0);
        float32x4_t vsub1 = vsubq_f32(vin3.val[1], vmean1);
        float32x4_t vsub2 = vsubq_f32(vin3.val[2], vmean2);
        float32x4_t vs0 = vmulq_f32(vsub0, vscale0);
        float32x4_t vs1 = vmulq_f32(vsub1, vscale1);
        float32x4_t vs2 = vmulq_f32(vsub2, vscale2);
        vst1q_f32(dout_c0, vs0);
        vst1q_f32(dout_c1, vs1);
        vst1q_f32(dout_c2, vs2);

        din += 12;
        dout_c0 += 4;
        dout_c1 += 4;
        dout_c2 += 4;
    }
    for (; i < size; i++) {
        *(dout_c0++) = (*(din++) - mean[0]) * scale[0];
        *(dout_c0++) = (*(din++) - mean[1]) * scale[1];
        *(dout_c0++) = (*(din++) - mean[2]) * scale[2];
    }
}
// 前处理部分没有改动
void pre_process(const cv::Mat& img, int width, int height, float* data) {
    cv::Mat rgb_img;
    cv::cvtColor(img, rgb_img, cv::COLOR_BGR2RGB);
    cv::resize(
        rgb_img, rgb_img, cv::Size(width, height), 0.f, 0.f, cv::INTER_CUBIC);
    cv::Mat imgf;
    rgb_img.convertTo(imgf, CV_32FC3, 1 / 255.f);
    std::vector<float> mean = { 0.485f, 0.456f, 0.406f };
    std::vector<float> scale = { 0.229f, 0.224f, 0.225f };
    const float* dimg = reinterpret_cast<const float*>(imgf.data);
    neon_mean_scale(dimg, data, width * height, mean, scale);
}
// 后处理部分
std::vector<Object> detect_object(const float* data,
    int count,
    float thresh,
    cv::Mat& image) {  // NOLINT
    if (data == nullptr) {
        std::cerr << "[ERROR] data can not be nullptr\n";
        exit(1);
    }
    std::vector<Object> rect_out;
    for (int iw = 0; iw < count; iw++) {
        int oriw = image.cols;
        int orih = image.rows;
        if (data[1] > thresh) {
            Object obj;
            int x = static_cast<int>(data[2]);
            int y = static_cast<int>(data[3]);
            int w = static_cast<int>(data[4] - data[2] + 1);
            int h = static_cast<int>(data[5] - data[3] + 1);
            cv::Rect rec_clip =
                cv::Rect(x, y, w, h) & cv::Rect(0, 0, image.cols, image.rows);
            obj.class_id = static_cast<int>(data[0]);
            obj.prob = data[1];
            obj.rec = rec_clip;
            // 改为只检测到未佩戴安全帽，才开始画框和报警
            if (w > 0 && h > 0 && obj.prob <= 1 && obj.class_id==0) {
                // 新增的报警函数
                warn();
                rect_out.push_back(obj);
                cv::rectangle(image, rec_clip, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
                std::string str_prob = std::to_string(obj.prob);
                std::string text = std::string(class_names[obj.class_id]) + ": " +
                    str_prob.substr(0, str_prob.find(".") + 4);
                int font_face = cv::FONT_HERSHEY_COMPLEX_SMALL;
                double font_scale = 1.f;
                int thickness = 1;
                cv::Size text_size =
                    cv::getTextSize(text, font_face, font_scale, thickness, nullptr);
                float new_font_scale = w * 0.5 * font_scale / text_size.width;
                text_size = cv::getTextSize(
                    text, font_face, new_font_scale, thickness, nullptr);
                cv::Point origin;
                origin.x = x + 3;
                origin.y = y + text_size.height + 3;
                cv::putText(image,
                    text,
                    origin,
                    font_face,
                    new_font_scale,
                    cv::Scalar(0, 255, 255),
                    thickness,
                    cv::LINE_AA);

                std::cout << "detection, image size: " << image.cols << ", "
                    << image.rows
                    << ", detect object: " << class_names[obj.class_id]
                    << ", score: " << obj.prob << ", location: x=" << x
                    << ", y=" << y << ", width=" << w << ", height=" << h
                    << std::endl;
            }
        }
        data += 6;
    }
    return rect_out;
}

void RunModel(std::string model_file, const cv::Mat& img) {
    // 1. Set MobileConfig
    MobileConfig config;
    config.set_model_from_file(model_file);

    // 2. Create PaddlePredictor by MobileConfig
    std::shared_ptr<PaddlePredictor> predictor =
        CreatePaddlePredictor<MobileConfig>(config);
    // 资源不足的话resize的图片一定要改小
    const int in_width = 320;
    const int in_height = 320;

    // 3. Prepare input data from image
    // input 0
    std::unique_ptr<Tensor> input_tensor0(std::move(predictor->GetInput(0)));
    input_tensor0->Resize({ 1, 3, in_height, in_width });
    auto* data0 = input_tensor0->mutable_data<float>();
    pre_process(img, in_width, in_height, data0);
    // input1
    std::unique_ptr<Tensor> input_tensor1(std::move(predictor->GetInput(1)));
    input_tensor1->Resize({ 1, 2 });
    auto* data1 = input_tensor1->mutable_data<int>();
    data1[0] = img.rows;
    data1[1] = img.cols;
    //开始计时
    char saveName[256];
    int start_time = get_current_us();

    // 4. Run predictor
    predictor->Run();
    //记录预测结束时间
    int end_time = get_current_us();
    //计算预测时间
    double process_time = (end_time - start_time) / 1000.0f;

    // 5. Get output and post process
    std::unique_ptr<const Tensor> output_tensor(
        std::move(predictor->GetOutput(0)));
    auto* outptr = output_tensor->data<float>();
    auto shape_out = output_tensor->shape();
    int64_t cnt = 1;
    for (auto& i : shape_out) {
        cnt *= i;
    }
    //注意这里，要从摄像头里把图片复制一份（画框用）
    cv::Mat output_image = img.clone();
    auto rec_out = detect_object(outptr, static_cast<int>(cnt / 6), 0.5f, output_image);
    // 把预测结束的时间戳作为文件名的一部分
    std::string name = std::to_string(end_time);
    std::string result_name =  name + "_yolov3_detection_result.jpg";
    // 截图存证
    cv::imwrite(result_name, output_image);
    printf("Preprocess time: %f ms\n", process_time);
}


int main(int argc, char** argv) {
    // 只需传入模型文件
    std::string model_file = argv[1];
    // 纯摄像头调用
    cv::VideoCapture cap(-1);
    cap.set(CV_CAP_PROP_FRAME_WIDTH, 640);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, 480);
    if (!cap.isOpened()) {
        return -1;
    }
    while (1) {
        cv::Mat input_image;
        cap >> input_image;
        RunModel(model_file, input_image);
        if (cv::waitKey(1) == char('q')) {
            break;
        }
    }
    cap.release();
    cv::destroyAllWindows();

    return 0;
}
```
## `CMakeLists.txt`文件
参考[Paddle-Lite-Demo/PaddleLite-armlinux-demo/object_detection_demo](https://gitee.com/paddlepaddle/Paddle-Lite-Demo/tree/master/PaddleLite-armlinux-demo/object_detection_demo)
```c++
cmake_minimum_required(VERSION 3.10)
set(CMAKE_SYSTEM_NAME Linux)
if(TARGET_ARCH_ABI STREQUAL "armv8")
    set(CMAKE_SYSTEM_PROCESSOR aarch64)
    set(CMAKE_C_COMPILER "aarch64-linux-gnu-gcc")
    set(CMAKE_CXX_COMPILER "aarch64-linux-gnu-g++")
elseif(TARGET_ARCH_ABI STREQUAL "armv7hf")
    set(CMAKE_SYSTEM_PROCESSOR arm)
    set(CMAKE_C_COMPILER "arm-linux-gnueabihf-gcc")
    set(CMAKE_CXX_COMPILER "arm-linux-gnueabihf-g++")
else()
    message(FATAL_ERROR "Unknown arch abi ${TARGET_ARCH_ABI}, only support armv8 and armv7hf.")
    return()
endif()

project(yolov3_detection)
message(STATUS "TARGET ARCH ABI: ${TARGET_ARCH_ABI}")
message(STATUS "PADDLE LITE DIR: ${PADDLE_LITE_DIR}")
include_directories(${PADDLE_LITE_DIR}/include)
link_directories(${PADDLE_LITE_DIR}/libs/${TARGET_ARCH_ABI})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
if(TARGET_ARCH_ABI STREQUAL "armv8")
    set(CMAKE_CXX_FLAGS "-march=armv8-a ${CMAKE_CXX_FLAGS}")
    set(CMAKE_C_FLAGS "-march=armv8-a ${CMAKE_C_FLAGS}")
elseif(TARGET_ARCH_ABI STREQUAL "armv7hf")
    set(CMAKE_CXX_FLAGS "-march=armv7-a -mfloat-abi=hard -mfpu=neon-vfpv4 ${CMAKE_CXX_FLAGS}")
    set(CMAKE_C_FLAGS "-march=armv7-a -mfloat-abi=hard -mfpu=neon-vfpv4 ${CMAKE_C_FLAGS}" )
endif()
find_package(OpenMP REQUIRED)
if(OpenMP_FOUND OR OpenMP_CXX_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    message(STATUS "Found OpenMP ${OpenMP_VERSION} ${OpenMP_CXX_VERSION}")
    message(STATUS "OpenMP C flags:  ${OpenMP_C_FLAGS}")
    message(STATUS "OpenMP CXX flags:  ${OpenMP_CXX_FLAGS}")
    message(STATUS "OpenMP OpenMP_CXX_LIB_NAMES:  ${OpenMP_CXX_LIB_NAMES}")
    message(STATUS "OpenMP OpenMP_CXX_LIBRARIES:  ${OpenMP_CXX_LIBRARIES}")
else()
    message(FATAL_ERROR "Could not found OpenMP!")
    return()
endif()
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND OR OpenCV_CXX_FOUND)
    include_directories(${OpenCV_INCLUDE_DIRS})
    message(STATUS "OpenCV library status:")
    message(STATUS "    version: ${OpenCV_VERSION}")
    message(STATUS "    libraries: ${OpenCV_LIBS}")
    message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
else()
    message(FATAL_ERROR "Could not found OpenCV!")
    return()
endif()
add_executable(yolov3_detection yolov3_detection.cc)
target_link_libraries(yolov3_detection paddle_light_api_shared ${OpenCV_LIBS})
target_link_libraries(yolov3_detection wiringPi)
```
## `build.sh`脚本
```bash
#!/bin/bash

# configure
TARGET_ARCH_ABI=armv8 # for RK3399, set to default arch abi
#TARGET_ARCH_ABI=armv7hf # for Raspberry Pi 3B
PADDLE_LITE_DIR=/home/pi/paddle-lite/build.lite.armlinux.armv8.gcc/inference_lite_lib.armlinux.armv8/cxx
if [ "x$1" != "x" ]; then
    TARGET_ARCH_ABI=$1
fi

# build
rm -rf build
mkdir build
cd build
cmake -DPADDLE_LITE_DIR=${PADDLE_LITE_DIR} -DTARGET_ARCH_ABI=${TARGET_ARCH_ABI} ..
make

# LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${PADDLE_LITE_DIR}/libs/ ./yolov3_detection ppyolo18.nb /home/pi/PaddleX/deploy/raspberry/hard_hat_workers0.jpg
```

# 部署效果
## 编译执行
```bash
pi@raspberrypi:~/paddle-lite/lite/demo/cxx/yolov3_detection $ sh build.sh 
-- The C compiler identification is GNU 8.3.0
-- The CXX compiler identification is GNU 8.3.0
-- Check for working C compiler: /usr/bin/aarch64-linux-gnu-gcc
-- Check for working C compiler: /usr/bin/aarch64-linux-gnu-gcc -- works
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Detecting C compile features
-- Detecting C compile features - done
-- Check for working CXX compiler: /usr/bin/aarch64-linux-gnu-g++
-- Check for working CXX compiler: /usr/bin/aarch64-linux-gnu-g++ -- works
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- TARGET ARCH ABI: armv8
-- PADDLE LITE DIR: /home/pi/paddle-lite/build.lite.armlinux.armv8.gcc/inference_lite_lib.armlinux.armv8/cxx
-- Found OpenMP_C: -fopenmp (found version "4.5") 
-- Found OpenMP_CXX: -fopenmp (found version "4.5") 
-- Found OpenMP: TRUE (found version "4.5")  
-- Found OpenMP  4.5
-- OpenMP C flags:  -fopenmp
-- OpenMP CXX flags:  -fopenmp
-- OpenMP OpenMP_CXX_LIB_NAMES:  gomp;pthread
-- OpenMP OpenMP_CXX_LIBRARIES:  /usr/lib/gcc/aarch64-linux-gnu/8/libgomp.so;/usr/lib/aarch64-linux-gnu/libpthread.so
-- Found OpenCV: /usr (found version "3.2.0") 
-- OpenCV library status:
--     version: 3.2.0
--     libraries: opencv_calib3d;opencv_core;opencv_features2d;opencv_flann;opencv_highgui;opencv_imgcodecs;opencv_imgproc;opencv_ml;opencv_objdetect;opencv_photo;opencv_shape;opencv_stitching;opencv_superres;opencv_video;opencv_videoio;opencv_videostab;opencv_viz;opencv_aruco;opencv_bgsegm;opencv_bioinspired;opencv_ccalib;opencv_datasets;opencv_dpm;opencv_face;opencv_freetype;opencv_fuzzy;opencv_hdf;opencv_line_descriptor;opencv_optflow;opencv_phase_unwrapping;opencv_plot;opencv_reg;opencv_rgbd;opencv_saliency;opencv_stereo;opencv_structured_light;opencv_surface_matching;opencv_text;opencv_ximgproc;opencv_xobjdetect;opencv_xphoto
--     include path: /usr/include;/usr/include/opencv
-- Configuring done
-- Generating done
-- Build files have been written to: /home/pi/paddle-lite/lite/demo/cxx/yolov3_detection/build
Scanning dependencies of target yolov3_detection
[ 50%] Building CXX object CMakeFiles/yolov3_detection.dir/yolov3_detection.cc.o
[100%] Linking CXX executable yolov3_detection
/usr/bin/ld: 当搜索用于 /usr/lib/gcc/aarch64-linux-gnu/8/../../../../lib/libwiringPi.so 时跳过不兼容的 -lwiringPi 
/usr/bin/ld: 当搜索用于 /usr/lib/../lib/libwiringPi.so 时跳过不兼容的 -lwiringPi 
/usr/bin/ld: 当搜索用于 /usr/lib/gcc/aarch64-linux-gnu/8/../../../libwiringPi.so 时跳过不兼容的 -lwiringPi 
[100%] Built target yolov3_detection
```
## 预测效果
```bash
pi@raspberrypi:~/paddle-lite/lite/demo/cxx/yolov3_detection $ ./build/yolov3_detection det_yolov3_mobilenetv3.nb 
[I 12/ 1 23:46:20.593 .../pi/paddle-lite/lite/core/device_info.cc:1064 Setup] ARM multiprocessors name: HARDWARE	: BCM2835
[I 12/ 1 23:46:20.593 .../pi/paddle-lite/lite/core/device_info.cc:1065 Setup] ARM multiprocessors number: 4
[I 12/ 1 23:46:20.593 .../pi/paddle-lite/lite/core/device_info.cc:1067 Setup] ARM multiprocessors ID: 0, max freq: 1200, min freq: 1200, cluster ID: 0, CPU ARCH: A53
[I 12/ 1 23:46:20.593 .../pi/paddle-lite/lite/core/device_info.cc:1067 Setup] ARM multiprocessors ID: 1, max freq: 1200, min freq: 1200, cluster ID: 0, CPU ARCH: A53
[I 12/ 1 23:46:20.593 .../pi/paddle-lite/lite/core/device_info.cc:1067 Setup] ARM multiprocessors ID: 2, max freq: 1200, min freq: 1200, cluster ID: 0, CPU ARCH: A53
[I 12/ 1 23:46:20.593 .../pi/paddle-lite/lite/core/device_info.cc:1067 Setup] ARM multiprocessors ID: 3, max freq: 1200, min freq: 1200, cluster ID: 0, CPU ARCH: A53
[I 12/ 1 23:46:20.593 .../pi/paddle-lite/lite/core/device_info.cc:1073 Setup] L1 DataCache size is: 
[I 12/ 1 23:46:20.593 .../pi/paddle-lite/lite/core/device_info.cc:1075 Setup] 32 KB
[I 12/ 1 23:46:20.593 .../pi/paddle-lite/lite/core/device_info.cc:1075 Setup] 32 KB
[I 12/ 1 23:46:20.593 .../pi/paddle-lite/lite/core/device_info.cc:1075 Setup] 32 KB
[I 12/ 1 23:46:20.593 .../pi/paddle-lite/lite/core/device_info.cc:1075 Setup] 32 KB
[I 12/ 1 23:46:20.593 .../pi/paddle-lite/lite/core/device_info.cc:1077 Setup] L2 Cache size is: 
[I 12/ 1 23:46:20.593 .../pi/paddle-lite/lite/core/device_info.cc:1079 Setup] 512 KB
[I 12/ 1 23:46:20.593 .../pi/paddle-lite/lite/core/device_info.cc:1079 Setup] 512 KB
[I 12/ 1 23:46:20.593 .../pi/paddle-lite/lite/core/device_info.cc:1079 Setup] 512 KB
[I 12/ 1 23:46:20.593 .../pi/paddle-lite/lite/core/device_info.cc:1079 Setup] 512 KB
[I 12/ 1 23:46:20.593 .../pi/paddle-lite/lite/core/device_info.cc:1081 Setup] L3 Cache size is: 
[I 12/ 1 23:46:20.594 .../pi/paddle-lite/lite/core/device_info.cc:1083 Setup] 0 KB
[I 12/ 1 23:46:20.594 .../pi/paddle-lite/lite/core/device_info.cc:1083 Setup] 0 KB
[I 12/ 1 23:46:20.594 .../pi/paddle-lite/lite/core/device_info.cc:1083 Setup] 0 KB
[I 12/ 1 23:46:20.594 .../pi/paddle-lite/lite/core/device_info.cc:1083 Setup] 0 KB
[I 12/ 1 23:46:20.594 .../pi/paddle-lite/lite/core/device_info.cc:1085 Setup] Total memory: 881860KB
detection, image size: 640, 480, detect object: head, score: 0.917295, location: x=155, y=25, width=128, height=164
Preprocess time: 8424.817383 ms
detection, image size: 640, 480, detect object: head, score: 0.899408, location: x=157, y=23, width=125, height=166
Preprocess time: 8228.533203 ms
detection, image size: 640, 480, detect object: head, score: 0.902913, location: x=156, y=25, width=126, height=161
Preprocess time: 8470.201172 ms
```
![file](https://ai-studio-static-online.cdn.bcebos.com/e1f650fbfb09406a9dc78fbe59a51ddd4125382b43824bdf8c7058dae616ce6f)


# 小结
<font size=4>仅就mAP这个评估指标而言，`yolov3_mobilenetv3`远胜于`ssd_mobilenet_v1`: 0.85+ v.s. 0.4+ </font>![https://ai-studio-static-online.cdn.bcebos.com/d500f483e55544a5b929abad59de208f180c068cc81648009fab60a0b6d9bda2](https://ai-studio-static-online.cdn.bcebos.com/d500f483e55544a5b929abad59de208f180c068cc81648009fab60a0b6d9bda2)

<font size=4>但预测时间这个指标就比较惨了，`yolov3_mobilenetv3`远远落后于`ssd_mobilenet_v1`: 8000+ ms v.s. 300 ms </font> ![http://finance.eastday.com/images/thumbnailimg/month_1809/a9a9c6d4e6d94555b7ad5152f421325d.png](http://finance.eastday.com/images/thumbnailimg/month_1809/a9a9c6d4e6d94555b7ad5152f421325d.png)

<font size=4>后续将继续关注ppyolo等其它模型的部署效果，敬请期待。</font>![https://ai-studio-static-online.cdn.bcebos.com/d500f483e55544a5b929abad59de208f180c068cc81648009fab60a0b6d9bda2](https://ai-studio-static-online.cdn.bcebos.com/d500f483e55544a5b929abad59de208f180c068cc81648009fab60a0b6d9bda2)
