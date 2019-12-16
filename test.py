
import os
os.environ['TFHUB_DOWNLOAD_PROGRESS'] = "1"
import tensorflow as tf
import tensorflow_hub as hub

# 官方模型
model = hub.KerasLayer('https://code.aliyun.com/qhduan/chinese_roberta_wwm_ext_L-12_H-768_A-12/raw/master/pool.tar.gz')

y = model([['我爱你']])
print(y)

