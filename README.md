# [在此处填写您的项目名称]

> 这是一个基于硬约束的自监督深度学习训练框架，旨在利用硬约束神经网络提升学习类算法在安全关键场景中的确定性和可靠性。

## 🛠️ 环境准备 (Installation)

在开始运行项目代码之前，请确保您的运行环境满足依赖要求。您可以通过以下命令一键安装所需库：

```bash
pip install -r requirements.txt
```


## 🚀 快速开始 (Quick Start)
本项目的主要运行流程严格遵循以下三个步骤：数据生成 -> 预训练 -> 硬约束训练。

1. 数据生成 (Data Generation)
首先，运行数据生成脚本以准备训练集和测试集。

默认路径：生成的据将自动保存在当前目录下的 dataset/ 文件夹中。

修改路径：如有需要，您可以在脚本中修改保存路径配置。

```bash
python data_generator.py
```
2. 模型预训练 (Pre-training)
数据准备完成后，使用预训练脚本对模型进行初始化。这一步旨在让模型学习基础特征，为后续的约束训练做准备。
```bash
python pre_train.py
```

3. 硬约束训练 (Hard-constrained Training)
最后，运行主训练脚本。此阶段将在预训练的基础上，进行带有硬约束 (Hard Constraints) 的模型优化。
```bash
python train.py
```

## 📂 文件结构说明
data_generator.py: 数据生成脚本

pre_train.py: 预训练脚本

train.py: 带有硬约束的主训练脚本

requirements.txt: 项目依赖环境列表

dataset/: 默认的数据存储目录（运行生成器后创建）
