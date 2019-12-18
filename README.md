# BERT Models

注达到本文效果基本要求Tensorflow 2.0

也许，是最简单的BERT预加载模型。

当然，实现起来是有一些tricky的，而且tokenizer并不是真正的bert的tokenizer，中文大部分不会有太大问题，英文的话实际上因为考虑BPE，所以肯定是不行的。

本项目重点在于，实际上我们是可以通过非常非常简单的几行代码，就能实现一个几乎达到SOTA的模型的。

## BERT分类模型（pool模式）

返回一个1x768的镜像，相当于句子的固定长度Embedding

根据一个实际Chinese GLUE的测试样例：[COLAB DEMO](https://colab.research.google.com/drive/1KkjPVn1s6_tSznhox5RxuKF9Igm8VAbE)

```python
import tensorflow_hub as hub

# 注意这里最后是 pool.tar.gz
model = hub.KerasLayer('https://code.aliyun.com/qhduan/chinese_roberta_wwm_ext_L-12_H-768_A-12/raw/master/pool.tar.gz')

# y.shape == (1, 768)
y = model([['我爱你']])
```

一个非常简单的样例（`classifier.py`）：

```python
import os
os.environ['TFHUB_DOWNLOAD_PROGRESS'] = "1"
import tensorflow as tf
import tensorflow_hub as hub

x = [
    ['我爱你'],
    ['我恨你'],
    ['爱你'],
    ['恨你'],
    ['爱'],
    ['恨'],
]
y = [
    1, 0, 1, 0, 1, 0
]

tx = tf.constant(x)
ty = tf.constant(tf.keras.utils.to_categorical(y, 2))

# 注意这里最后是 pool.tar.gz
model = tf.keras.Sequential([
  hub.KerasLayer('https://code.aliyun.com/qhduan/chinese_roberta_wwm_ext_L-12_H-768_A-12/raw/master/pool.tar.gz', trainable=False),
  tf.keras.layers.Dense(2, activation='softmax')
])

model.compile(loss='categorical_crossentropy')
model.fit(tx, ty, epochs=10, batch_size=2)
logits = model.predict(tx)
pred = logits.argmax(-1).tolist()

print(pred)
print(y)
```

## BERT序列模型（SEQ）

返回一个序列的Embedding的模型

```python
import tensorflow_hub as hub

# 注意这里最后是 seq.tar.gz
model = hub.KerasLayer('https://code.aliyun.com/qhduan/chinese_roberta_wwm_ext_L-12_H-768_A-12/raw/master/seq.tar.gz')

# y.shape == (1, 5, 768)
# [CLS], 我, 爱, 你, [SEP]，所以一共5个符号
y = model([['我爱你']])
```

## BERT预测模型（PRED）

例如使用mask预测缺字

```python
import tensorflow_hub as hub

# 注意这里最后是 pred.tar.gz
model = hub.KerasLayer('https://code.aliyun.com/qhduan/chinese_roberta_wwm_ext_L-12_H-768_A-12/raw/master/pred.tar.gz')

# y.shape == (1, 5, 21128)
y = model([['我[MASK]你']])

index2word = {k: v.strip() for k, v in enumerate(open('vocab.txt'))}

# 我 爱 你
r = [index2word[i] for i in y.numpy().argmax(-1).flatten()][1:-1]
```


## 模型引用

Roberta和WMM来自[ymcui](https://github.com/ymcui/Chinese-BERT-wwm)
