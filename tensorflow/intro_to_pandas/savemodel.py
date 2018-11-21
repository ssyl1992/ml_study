import tensorflow as tf
import numpy as np


def train_model():


    # prepare the data
    x_data = np.random.rand(100).astype(np.float32)
    print (x_data)
    y_data = x_data * 0.1 + 0.2
    print (y_data)

    # define the weights
    W = tf.Variable(tf.random_uniform([1], -20.0, 20.0), dtype=tf.float32, name='w')
    b = tf.Variable(tf.random_uniform([1], -10.0, 10.0), dtype=tf.float32, name='b')
    y = W * x_data + b

    # define the loss
    loss = tf.reduce_mean(tf.square(y - y_data))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    # save model

    saver = tf.train.Saver(max_to_keep=4)
    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        print ("------------------------------------------------------")
        print ("before the train, the W is %6f, the b is %6f" % (sess.run(W), sess.run(b)))
        sess.run(loss)
        for epoch in range(300):
            if epoch % 10 == 0:

                print ("after epoch %d, the loss is %6f" % (epoch, sess.run(loss)))
                print ("the W is %f, the b is %f" % (sess.run(W), sess.run(b)))
                saver.save(sess, "model/my-model", global_step=epoch)
                print ("save the model")
            sess.run(train_step)
        print ("------------------------------------------------------")

def load_model():
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('model/my-model-290.meta')
        saver.restore(sess, tf.train.latest_checkpoint("model/"))
        print (sess.run('w:0'))
        print (sess.run('b:0'))

train_model()
load_model()

