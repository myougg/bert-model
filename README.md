# BERT Models

也许，是最简单的BERT预加载模型。

当然，实现起来是有一些tricky的，而且tokenizer并不是真正的bert的tokenizer，中文大部分不会有太大问题，英文的话实际上因为考虑BPE，所以肯定是不行的。

本项目重点在于，实际上我们是可以通过非常非常简单的几行代码，就能实现一个几乎达到SOTA的模型的。

## BERT分类模型

返回一个1x768的镜像

根据一个实际Chinese GLUE的测试样例：[COLAB DEMO](https://colab.research.google.com/drive/1KkjPVn1s6_tSznhox5RxuKF9Igm8VAbE)

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

model = tf.keras.Sequential([
  hub.KerasLayer('https://code.aliyun.com/qhduan/bert-pool/raw/master/bert_simple_tokenizer.tar.gz', trainable=False),
  hub.KerasLayer('https://code.aliyun.com/qhduan/bert-pool/raw/master/bert_pool_chinese_roberta_wwm_ext_L-12_H-768_A-12.tar.gz', trainable=False),
  tf.keras.layers.Dense(2, activation='softmax')
])

model.compile(loss='categorical_crossentropy')
model.fit(tx, ty, epochs=10, batch_size=2)
logits = model.predict(tx)
pred = logits.argmax(-1).tolist()

print(pred)
print(y)
```

## BERT序列模型

返回一个序列的Embedding的模型，类似Bert-as-A-Service

```python
import tensorflow as tf
import tensorflow_hub as hub


# 官方模型
tokenizer = hub.KerasLayer('https://code.aliyun.com/qhduan/bert/raw/master/bert_simple_tokenizer.tar.gz')
model = hub.KerasLayer('https://code.aliyun.com/qhduan/bert/raw/master/bert_chinese_L-12_H-768_A-12.tar.gz')

x = tf.constant([['我爱你']])
ids = tokenizer(x)
y = model(ids)
```

其他模型：

来自[ymcui](https://github.com/ymcui/Chinese-BERT-wwm)

`https://code.aliyun.com/qhduan/bert/raw/master/bert_chinese_wwm_ext_L-12_H-768_A-12.tar.gz`

`https://code.aliyun.com/qhduan/bert/raw/master/bert_chinese_roberta_wwm_ext_L-12_H-768_A-12.tar.gz`

