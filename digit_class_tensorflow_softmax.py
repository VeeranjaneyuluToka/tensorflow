# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 16:12:49 2017

@author: DELL
"""

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

def main():
    mnist = input_data.read_data_sets("tmp/data/", one_hot=True)
    
    sess = tf.InteractiveSession()
    
    X = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])
    
    #print("X:",X.get_shape())
    #print("y_",y_.get_shape())
    
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    
    #print("W:",W.get_shape())
    #print("b:",b.get_shape())
    
    sess.run(tf.global_variables_initializer())
    
    #predict class and loss function
    y = tf.matmul(X, W) + b
    #print("y:",y.get_shape())
    
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    
    #train the model
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    
    for _ in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
    #    print(np.shape(batch_xs))
    #    print(np.shape(batch_ys))
        sess.run(train_step, feed_dict={X:batch_xs, y_:batch_ys})
        
    #evaluate the model
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    print(accuracy.eval(feed_dict={X:mnist.test.images, y_:mnist.test.labels}))

if __name__ == '__main__':
    main()
