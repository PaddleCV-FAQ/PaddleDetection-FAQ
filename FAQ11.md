#### 如何选择用来预测的模型文件
**例：**使用名为 `model_final` 的模型文件进行预测

- 方法1：
  在`./configs/ssdlite_mobilenet_v3_small.yml` 中， `weights` 参数设定为 `output/ssdlite_mobilenet_v3_small/model_final`
  ![](https://ai-studio-static-online.cdn.bcebos.com/02c3521893314236b0f6bbc6066d751d6451313e0ce146838ff689d7366f675d)

  

- 方法2：
执行命令后添加 `-o weights=output/ssdlite_mobilenet_v3_small/model_final`
![](https://ai-studio-static-online.cdn.bcebos.com/289b0ef2f47b4225af17ab95b64a17ac1c312c49fa544c798c841d00205e245f)