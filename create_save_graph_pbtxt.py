import tensorflow as tf
import numpy as np
print(tf.__version__)
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

W=tf.Variable(initial_value=tf.random_normal([1]), name='weight',trainable=True)
b=tf.Variable(initial_value=0.001,name='bias',trainable=True)

x=tf.placeholder(dtype=tf.float32, shape=[1],name='x')
y=tf.add(tf.multiply(W,x),b,name='output')
init=tf.global_variables_initializer()
saver=tf.train.Saver()
save_path="/home/checkpoint/" #path in google colab
model_save=save_path+"model.ckpt"

#TensorFlow session
with tf.Session() as sess:
     #initialising the variables
    sess.run(init)
    #sample run(optional)
    op=sess.run(y, feed_dict={x: np.reshape(1.5,[1])}) 
    #saving the model
    saver.save(sess,model_save) 
    #saving the model's tensorflow graph definition
    tf.train.write_graph(sess.graph_def, save_path, 'savegraph.pbtxt') 