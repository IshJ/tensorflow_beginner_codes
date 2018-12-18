import tensorflow as tf
import numpy as np
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt

learning_rate = 0.01
epoch_count = 1000
cost_trace = []


def read_infile():
 data = load_boston()
 features = np.array(data.data)
 target = np.array(data.target)
 return features,target

def normalize(features):
#   mn = tf.reduce_sum(features,0)/features.shape[0]
#   std = tf.math.sqrt(tf.reduce_mean(tf.pow((features-mn),2))/features.shape[0])
  mn = np.mean(features,axis = 0)
  std = np.std(features,axis = 0)
  
  
  features = (features-mn)/std
  return features


features,target = read_infile()



num_features = features.shape[1]
size = features.shape[0]

features = normalize(features)
features = features.reshape(-1,num_features )
target = target.reshape(-1,1)

x = tf.placeholder(tf.float32,(None, num_features))
y = tf.placeholder(tf.float32,(None, 1))

a = tf.Variable(tf.random_normal((1, num_features)), name  = 'weights')
b = tf.Variable(0.00001)

pred = x*a+b

error = pred - y
cost = tf.reduce_mean(tf.square(error))
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init_op =  tf.global_variables_initializer()

sess= tf.Session()
sess.run(init_op)
for _ in range(epoch_count):
  sess.run(train_op, feed_dict={x : features, y : target})  
  cost_trace.append(sess.run(cost, feed_dict={x : features, y : target}) )
sess.close()

print 'MSE in training:',cost_trace[-1]
plt.plot(cost_trace) 
                           
                           
                      