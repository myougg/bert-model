
import os
os.environ['TFHUB_DOWNLOAD_PROGRESS'] = "1"
import tensorflow as tf
import tensorflow_hub as hub

# 官方模型
tokenizer = hub.KerasLayer('https://code.aliyun.com/qhduan/bert/raw/master/bert_simple_tokenizer.tar.gz')
model = hub.KerasLayer('https://code.aliyun.com/qhduan/bert/raw/master/bert_chinese_L-12_H-768_A-12.tar.gz')

x = tf.constant([['我爱你']])
ids = tokenizer(x)
y = model(ids)
print(y)

