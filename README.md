# bert_slot_tagging
用预训练BERT实现序列标注模型。

## 依赖环境
- python==3.6.5
- allennlp==0.9.0
- torch==1.3.1

## 运行

### 训练atis
```
python3 train.py --config_path ./config/bert.atis.json --output_dir ./output/bert-atis/
```

### 预测atis
```
python3 test.py --output_dir ./output/bert-atis/
```

## 数据格式说明

- 放在一个文件夹下，分成两个文件seq.in和seq.out
- seq.in中存放输入序列，每行一个样本，序列token用空格分隔
- seq.out中存放输出序列（BIO标注），每行一个样本，序列token用空格分隔
- seq.in和seq.out中的每一行（每个样本）需要一一对应。

## 配置文件说明

![配置说明1](https://raw.githubusercontent.com/yym6472/ImagesForPicGo/master/20200325163112.png)

![配置说明2](https://raw.githubusercontent.com/yym6472/ImagesForPicGo/master/%24VN%6068PRW7LA%7DFZOPLFB4QJ.png)