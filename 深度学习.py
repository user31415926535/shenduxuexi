import tensorflow as tf
import torch
#一.创建张量

#value: 定义张量的n维数组值，可选
#dtype: 定义张量数据类型，例如：
#tf.string: 字符串类型
#tf.float32: 浮点类型
#tf.int16: 整型
#“name”: 张量的名字，可选，默认：“Const_1:0”(疑问：这里输出并没有显示名称而是显示了维度)
r1 = tf.constant( 1 ,tf.int16, name="my_scalar") 
print(r1)
#"Const:0" – 张量名称
#shape – 张量形状
#dtype – 张量数据类型
r1_vector = tf.constant([1,3,5], tf.int16)
print(r1_vector)
r2_boolean = tf.constant([True, True, False], tf.bool)
print(r2_boolean)   

#二.张量形状

#创建元素为0，形状为（10）的张量
print(tf.zeros(10))
#创建元素为1，形状为（10）的张量
print(tf.ones(10))
#创建m_shape，r4。使r4列数与m_shape相同并元素为1,m_shape.shape后[1]使r4列数相同，[0]使r4行数相同
m_shape=tf.constant([[10,11],
                     [12,13],
                     [14,15]])
r4=tf.ones(m_shape.shape[1])
print(r4)





print(m_shape)






