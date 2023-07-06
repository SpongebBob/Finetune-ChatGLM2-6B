# Finetune-ChatGLM2-6B
ChatGLM2-6B 全参数微调，支持多轮对话的高效微调。


# ChatGLM2-6B-全参数微调
本仓库实现了对于 ChatGLM2-6B 模型的全参数微调，主要改进点在于支持了ChatGLM2-6B 多轮对话的全参数微调，在多轮对话的交互组织方面，使用了更高效的session级别高效训练，训练效果相比原版ChatGLM-6B有较大提升。 

Demo数据方面使用了开源项目[BELLE](https://github.com/LianjiaTech/BELLE/tree/main/train)提供的数据。

## 软件依赖
运行微调需要4.27.1版本的`transformers`。除 ChatGLM-6B 的依赖之外，还需要安装以下依赖
```
pip install rouge_chinese nltk jieba datasets
```
## 数据方面

### 下载数据

```bash
wget https://huggingface.co/datasets/BelleGroup/school_math_0.25M/resolve/main/school_math_0.25M.json
```

该数据是数学应用题求解任务，包含解题过程，共计约25万条。示例：

| Instruction                                                                                     | Answer                                                                                                                                                                                                                                  |
| ----------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 一箱鸡蛋有25个，小明爸爸每天给他9个鸡蛋吃，那么一箱鸡蛋能够供小明吃几天？                       | 一箱鸡蛋有25个，小明每天吃9个鸡蛋，用除法计算一下即可： 25 ÷ 9 = 2余7 也就是说，小明第1天到第2天共吃2个鸡蛋，第3天吃剩下的7个鸡蛋。 因此，一箱鸡蛋能够供小明吃2天。                                                                    |
| 小明手中有15颗糖果，他想要分给他的朋友小红和小李，每人分得的糖果数是3颗，那么他还剩下几颗糖果？ | 该题可以用减法来解决。小明手中有15颗糖果，每人分得3颗，所以小红和小李一共分得6颗糖果（3+3=6）。 2. 然后再用原来的糖果总数减去分给小红和小李的糖果数，即可得到小明手中剩余的糖果数。 计算过程如下： 15 - 6 = 9 所以，小明还剩下9颗糖果。 |

### 转换数据格式

```bash
python convert_to_conv_data.py --orig_data school_math_0.25M.json --write_data school_math_0.25M_conv.json --dataset_name bellemath
head -n 1000 school_math_0.25M_conv.json > belleMath-dev1K.json
tail -n +1001 school_math_0.25M_conv.json > belleMath.json
```

demo选取前1000条作为验证集，其余数据作为训练集。

**注：**目前仓库的代码要兼容单轮和多轮数据，因此对于之前的开源数据(格式为[instruction,output]，目前绝大部分开源的单轮instruction数据均为这种格式)，需要用convert_to_conv_data.py脚本转换为如下的统一格式:

```bash
{"id": "bellemath42", "conversations": [{"from": "human", "value": " 一箱鸡蛋有25个，小明爸爸每天给他9个鸡蛋吃，那么一箱鸡蛋能够供小明吃几天？"}, {"from": "assistant", "value": "\n一箱鸡蛋有25个，小明每天吃9个鸡蛋，用除法计算一下即可：\n25 ÷ 9 = 2余7\n也就是说，小明第1天到第2天共吃2个鸡蛋，第3天吃剩下的7个鸡蛋。\n因此，一箱鸡蛋能够供小明吃2天。"}]}
{"id": "bellemath80", "conversations": [{"from": "human", "value": "小明手中有15颗糖果，他想要分给他的朋友小红和小李，每人分得的糖果数是3颗，那么他还剩下几颗糖果？"}, {"from": "assistant", "value": " \n1. 该题可以用减法来解决。小明手中有15颗糖果，每人分得3颗，所以小红和小李一共分得6颗糖果（3+3=6）。\n2. 然后再用原来的糖果总数减去分给小红和小李的糖果数，即可得到小明手中剩余的糖果数。 \n计算过程如下：\n15 - 6 = 9\n所以，小明还剩下9颗糖果。"}]}
```

其他的训练数据见：https://huggingface.co/BelleGroup  按照上述流程转换格式即可。

对于多轮对话数据， [shareGPT](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/tree/main) 是一个开源的大规模多轮对话数据，当前代码已支持训练这种多轮对话数据。数据下载：

```bash
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
```


### 训练


#### Finetune

需要安装 [Deepspeed](https://github.com/microsoft/DeepSpeed)，修改model_name_or_path为下载好的chatglm2-6b的路径，然后运行以下指令：

```shell
bash ds_train_finetune.sh
```

### 推理


## 模型部署
首先载入Tokenizer：

```python
from transformers import AutoConfig, AutoModel, AutoTokenizer

# 载入Tokenizer
tokenizer = AutoTokenizer.from_pretrained("download_path", trust_remote_code=True)
```

加载的是新 Checkpoint：

```python

model = AutoModel.from_pretrained("output/checkpoint", trust_remote_code=True)
```

之后根据需求可以进行量化，也可以直接使用：

```python
# Comment out the following line if you don't use quantization
model = model.quantize(4)
model = model.half().cuda()
model.transformer.prefix_encoder.float()
model = model.eval()

response, history = model.chat(tokenizer, "你好", history=[])
```



## 使用自己的数据集
参考demo格式，修改 `train.sh` 和 `evaluate.sh` 中的 `train_file`、`validation_file`和`test_file`为你自己的 JSON 格式数据集路径。可能还需要增大  `max_length` 来匹配你自己的数据集中的最大输入输出长度。参考自有多轮对话数据训练loss
![image](https://github.com/SpongebBob/Finetune-ChatGLM2-6B/assets/11678997/736cd1d7-4ffc-49c2-a64e-d7f7991802c9)




## 引用

```
1.https://github.com/THUDM/ChatGLM-6B
2.https://github.com/LianjiaTech/BELLE
3.https://github.com/THUDM/ChatGLM2-6B
```
