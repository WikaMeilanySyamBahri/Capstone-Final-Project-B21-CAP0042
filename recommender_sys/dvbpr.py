import sys
import math
import random
import time
from PIL import Image
import queue
import numpy as np
import threading
from io import StringIO

#set the hyperparameters
K = 100
learning_rate = 1e-4
training_epoch = 20
batch_size = 128
lambda1 = 0.001
lambda2 = 1.0
dropout = 0.5
numldprocess = 4

#load dataset
dataset_name = 'AmazonFashion6ImgPartitioned.npy'
dataset = np.load('../'+dataset_name)
[user_train, user_validation, user_test, Item, usernum, itemnum] = dataset

def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')

def avgpool2d(x, k=2):
    return tf.nn.avg_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')

#filling the weights from the research
weights = {
    'wc1': [11, 11, 3, 64],
    'wc2': [5, 5, 64, 256],
    'wc3': [3, 3, 256, 256],
    'wc4': [3, 3, 256, 256],
    'wc5': [3, 3, 256, 256],
    'wd1': [7*7*256, 4096],
    'wd2': [4096, 4096],
    'wd3': [4096, K],
}

#defining the weights
def Weights(name):
    return tf.get_variable(name,dtype=tf.float32,shape=weights[name],
                           initializer=tf.contrib.layers.xavier_initializer())

#filling the biases for each layers
biases = {
    'bc1': [64],
    'bc2': [256],
    'bc3': [256],
    'bc4': [256],
    'bc5': [256],
    'bd1': [4096],
    'bd2': [4096],
    'bd3': [K],
}

def Biases(name):
    return tf.get_variable(name,dtype=tf.float32,initializer=tf.zeros(biases[name]))

# Initializing the variables
init = tf.initialize_all_variables()

def AUC(train,test,U,I):
    ans=0
    cc=0
    for u in train:
        i=test[u][0]['productid']
        T=np.dot(U[u,:],I.T)
        cc+=1
        M=set()
        for item in train[u]:
            M.add(item['productid'])
        M.add(i)

        count=0
        tmpans=0
        #for j in xrange(itemnum):
        for j in random.sample(xrange(itemnum),100): #sample
            if j in M: continue
            if T[i]>T[j]: tmpans+=1
            count+=1
        tmpans/=float(count)
        ans+=tmpans
    ans/=float(cc)
    return ans

def Evaluation(step):
    print '...'
    U=sess.run(thetau)
    I=np.zeros([itemnum,K],dtype=np.float32)
    idx=np.array_split(range(itemnum),(itemnum+batch_size-1)/batch_size)

    input_images=np.zeros([batch_size,224,224,3],dtype=np.int8)
    for i in range(len(idx)):
        cc=0
        for j in idx[i]:
            input_images[cc]=np.uint8(np.asarray(Image.open(StringIO(Item[j]['imgs'])).convert('RGB').resize((224,224))))
            cc+=1
        I[idx[i][0]:(idx[i][-1]+1)]=sess.run(result_test,feed_dict={image_test:input_images})[:(idx[i][-1]-idx[i][0]+1)]
    print 'export finised!'
    np.save('UI_'+str(K)+'_'+str(step)+'.npy',[U,I])
    return AUC(user_train,user_validation,U,I), AUC(user_train,user_test,U,I)

def sample(user):
    u = random.randrange(usernum)
    numu = len(user[u])
    i = user[u][random.randrange(numu)]['productid']
    M=set()
    for item in user[u]:
        M.add(item['productid'])
    while True:
        j=random.randrange(itemnum)
        if (not j in M): break
    return (u,i,j)

def load_image_async():
    while True:
        (uuu,iii,jjj)=sample(user_train)
        jpg1=np.uint8(np.asarray(Image.open(StringIO(Item[iii]['imgs'])).convert('RGB').resize((224,224))))
        jpg2=np.uint8(np.asarray(Image.open(StringIO(Item[jjj]['imgs'])).convert('RGB').resize((224,224))))
        sess.run(batch_train_queue_op,feed_dict={queueu:np.asarray([uuu]),
                                                 queuei:np.asarray([iii]),
                                                 queuej:np.asarray([jjj]),
                                                 queueimage1:jpg1,queueimage2:jpg2,
                                                })

f=open('DVBPR.log','w')
config = tf.ConfigProto(log_device_placement=False,allow_soft_placement=True)
sess=tf.Session(config=config)
sess.run(init)

t=[0]*numldprocess
for i in range(numldprocess):
    t[i] = threading.Thread(target=load_image_async)
    t[i].daemon=True
    t[i].start()

oneiteration = 0
for item in user_train: oneiteration+=len(user_train[item])

step = 1
saver = tf.train.Saver([k for k in tf.global_variables() if k.name.startswith('DVBPR')])

epoch=0
while step * batch_size <= training_epoch*oneiteration+1:

    sess.run(optimizer, feed_dict={keep_prob: dropout})

    print 'Step#'+str(step)+' CNN update'

    if step*batch_size / oneiteration >epoch:
        epoch+=1
        saver.save(sess,'./DVBPR_auc_'+str(K)+'_'+str(step)+'.ckpt')
        auc_valid,auc_test=Evaluation(step)
        print 'Epoch #'+str(epoch)+':'+str(auc_test)+' '+str(auc_valid)+'\n'
        f.write('Epoch #'+str(epoch)+':'+str(auc_test)+' '+str(auc_valid)+'\n')
        f.flush()
    step += 1
print "DVBPR ready"
