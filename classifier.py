
import os
os.environ['TFHUB_DOWNLOAD_PROGRESS'] = "1"
import tensorflow as tf
import tensorflow_hub as hub

# 官方模型
tokenizer = hub.KerasLayer('https://code.aliyun.com/qhduan/bert/raw/master/bert_simple_tokenizer.tar.gz')
bert = hub.KerasLayer('https://code.aliyun.com/qhduan/bert/raw/master/bert_chinese_L-12_H-768_A-12.tar.gz')

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
    tokenizer,
    bert,
    tf.keras.layers.Lambda(lambda x: x[:, 0, :]),
    tf.keras.layers.Dense(2, activation='softmax')
])
model.compile(loss='mse')
model.fit(tx, ty, epochs=10, batch_size=2)
logits = model.predict(tx)
pred = logits.argmax(-1).tolist()

print(pred)
print(y)

