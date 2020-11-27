#### 5.对于模型的结构了解不清晰，对于加载与训练模型的一些warning有疑问

- ##### 模型详解：
  保存模型有5个API，分别是`fluid.io.save_vars`、`fluid.io.save_params`、`fluid.io.save_persistables`、`paddle.fluid.save`、`fluid.io.save_inference_model`。
          其中可用于 **`checkpoint` **的模型保存一般用 `fluid.io.save_vars`、`fluid.io.save_params` ，官方提供的预训练模型大多为 `fluid.io.save_params` 保存的，有参数 `filename` 可指定是否分开保存各变量。
          用于**推理**的模型用 `fluid.io.save_inference_model`保存使用。

  ​       `fluid.io.save_persistables`、`paddle.fluid.save`保存的模型需要删掉优化器和学习率才可以使用，前者有参数 `filename` 可指定是否分开保存各变量，后者只能保存以 `.pdparams`、`.pdmodel`、`.pdopt`为后缀的模型。

  ​        - 所有API都是默认散装保存的

- ##### 产出模型：
1. 通过 `./tools/train.py` 训练出来的模型（指定 `filename=model_final` ,整合保存）

```
./output/XXXXX/
	model_final.pdparams   网络参数
	model_final.pdmodel    训练模型
	model_final.pdopt      输入输出变量
```
2. 通过 `./tools/export_model.py` 转化的模型
```
./inference_model_final/XXXXX/
	__model__      模型
	__params__     参数
	infer_cfg.yml  变量
```
- ##### 加载模型：
1. 通过 `./tools/infer.py` 进行预测
加载的模型为原生模型：`model_final.pdparams、model_final.pdmodel、model_final.pdopt`，XXXXX.yml中的 weights参数可定义加载模型路径及名称。
2. 通过 `./deploy/python/infer.py` 进行预测
加载的模型为经过 `./tools/export_model.py` 转化的预测模型：`__model__、__params__、infer_cfg.yml`
- ##### 关于warning
版本升级后，api参数有变化，可指定保存的所有变量文件，有两种形式，一种是**所有变量会按照变量名称单独保存**成文件，另一种是**所有变量会保存成一个文件**名为该设置值的文件（**注：**结构不同，使用时一样）。这两种都是可以进行继续训练的节点模型，同时加载这两种的中的一种做为预训练模型，在加载预训练模型时会出现 `not found`、 `not used`的警告情况，这样的warning不影响训练，可以忽略。

