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

## 天池TODO

- ~~写一个和比赛一样的评测指标，作为early-stop的指标~~
- ~~加入规则（根据训练集的词典硬匹配，可以适当提高召回率，鼓励模型多预测），和模型预测的结果融合~~
- 对对抗攻击的方法调参、尝试其它的数据增强方法（例如：EDA）
- ~~bert/roberta预训练模型可更新~~
- 尝试其它的预训练模型（ALBERT、T5等）
- 尝试clue的roberta模型
- 尝试加入词典
- ERROR Case study, 查看是什么原因引起的错误（即准确率不高）
- 使用模型ensemble（修改成每个模型进行一次预测缓存起来）
- 可供参考的tricks：https://github.com/ChileWang0228/Deep-Learning-With-Python/tree/master/chapter8
- chinese-roberta-wwm预训练模型说明：https://github.com/ymcui/Chinese-BERT-wwm#%E5%BF%AB%E9%80%9F%E5%8A%A0%E8%BD%BD