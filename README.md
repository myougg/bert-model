# BERT Models

也许，是最简单的BERT预加载模型。

当然，实现起来是有一些tricky的，而且tokenizer并不是真正的bert的tokenizer，中文大部分不会有太大问题，英文的话实际上因为考虑BPE，所以肯定是不行的。

```python
import tensorflow as tf
import tensorflow_hub as hub


tokenizer = hub.KerasLayer('https://code.aliyun.com/qhduan/bert-model/raw/bert_simple/bert_simple_tokenizer.tar.gz')
model = hub.KerasLayer('https://code.aliyun.com/qhduan/bert-model/raw/bert_simple/bert_chinese_L-12_H-768_A-12.tar.gz')

x = tf.ragged.constant([list('我爱你')]).to_tensor('[PAD]')
ids = tokenizer(x)
y = model(ids)
```
