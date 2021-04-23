
"""
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                        %
%                Create a CNN in tensorflow (CIFAR dataset)              %
%                                                                        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  Summary: This script is used to create a CNN model to classify CIFAR dataset

 Credits: This script is made possible due to the guidance from tan_nguyen 
 the TA for the introduction to deep learning course at the Rice University 
 Author: Akshay Sujatha Ravindran
 email: akshay dot s dot ravindran at gmail dot com
 Dec 23rd 2018
"""

from scipy import misc
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
from math import sqrt


def put_kernels_on_grid (kernel, pad = 1):

  '''Visualize conv. filters as an image (mostly for the 1st layer).
  Arranges filters into a grid, with some paddings between adjacent filters.

  Args:
    kernel:            tensor of shape [Y, X, NumChannels, NumKernels]
    pad:               number of black pixels around each filter (between them)

  Return:
    Tensor of shape [1, (Y+2*pad)*grid_Y, (X+2*pad)*grid_X, NumChannels].
  '''
  # get shape of the grid. NumKernels == grid_Y * grid_X
  def factorization(n):
    for i in range(int(sqrt(float(n))), 0, -1):
      if n % i == 0:
        if i == 1: print('Who would enter a prime number of filters')
        return (i, int(n / i))
  (grid_Y, grid_X) = factorization (kernel.get_shape()[3].value)
  print ('grid: %d = (%d, %d)' % (kernel.get_shape()[3].value, grid_Y, grid_X))

  x_min = tf.reduce_min(kernel)
  x_max = tf.reduce_max(kernel)
  kernel = (kernel - x_min) / (x_max - x_min)

  # pad X and Y
  x = tf.pad(kernel, tf.constant( [[pad,pad],[pad, pad],[0,0],[0,0]] ), mode = 'CONSTANT')

  # X and Y dimensions, w.r.t. padding
  Y = kernel.get_shape()[0] + 2 * pad
  X = kernel.get_shape()[1] + 2 * pad

  channels = kernel.get_shape()[2]

  # put NumKernels to the 1st dimension
  x = tf.transpose(x, (3, 0, 1, 2))
  # organize grid on Y axis
  x = tf.reshape(x, tf.stack([grid_X, Y * grid_Y, X, channels]))

  # switch X and Y axes
  x = tf.transpose(x, (0, 2, 1, 3))
  # organize grid on X axis
  x = tf.reshape(x, tf.stack([1, X * grid_X, Y * grid_Y, channels]))

  # back to normal order (not combining with the next step for clarity)
  x = tf.transpose(x, (2, 1, 3, 0))

  # to tf.image_summary order [batch_size, height, width, channels],
  #   where in this case batch_size == 1
  x = tf.transpose(x, (3, 0, 1, 2))

  # scaling to [0, 255] is not necessary for tensorboard
  return x

# --------------------------------------------------
# setup
 
def weight_variable(shape):
    '''
    Initialize weights
    :param shape: shape of weights, e.g. [w, h ,Cin, Cout] where
    w: width of the filters
    h: height of the filters
    Cin: the number of the channels of the filters
    Cout: the number of filters
    :return: a tensor variable for weights with initial values
    '''

    # IMPLEMENT YOUR WEIGHT_VARIABLE HERE
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)

    #return W

def bias_variable(shape):
    '''
    Initialize biases
    :param shape: shape of biases, e.g. [Cout] where
    Cout: the number of filters
    :return: a tensor variable for biases with initial values
    '''

    # IMPLEMENT YOUR BIAS_VARIABLE HERE
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

    #return b

def conv2d(x, W):
    '''
    Perform 2-D convolution
    :param x: input tensor of size [N, W, H, Cin] where
    N: the number of images
    W: width of images
    H: height of images
    Cin: the number of channels of images
    :param W: weight tensor [w, h, Cin, Cout]
    w: width of the filters
    h: height of the filters
    Cin: the number of the channels of the filters = the number of channels of images
    Cout: the number of filters
    :return: a tensor of features extracted by the filters, a.k.a. the results after convolution
    '''

    # IMPLEMENT YOUR CONV2D HERE
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    #return h_conv

def max_pool_2x2(x):
    '''
    Perform non-overlapping 2-D maxpooling on 2x2 regions in the input data
    :param x: input data
    :return: the results of maxpooling (max-marginalized + downsampling)
    '''

    # IMPLEMENT YOUR MAX_POOL_2X2 HERE
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    #return h_max


def create_summaries(var, name):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean_' + name, mean)
    stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
    tf.summary.scalar('stddev_' + name, stddev)
    tf.summary.scalar('max_' + name, tf.reduce_max(var))
    tf.summary.scalar('min_' + name, tf.reduce_min(var))
    tf.summary.histogram('y_' + name, var)
    return None

def Optimisation(X,opttype,LR,mom):
    if opttype=="Adam":
        OUT=tf.train.AdamOptimizer(LR).minimize(X)  
    elif opttype=="Adagrad":
       OUT= tf.train.AdagradOptimizer(LR).minimize(X)    
    elif opttype=="Momentum":
        
        OUT=tf.train.MomentumOptimizer(LR,mom).minimize(X)
    elif opttype=="RMSprop":    

        OUT=tf.train.RMSPropOptimizer(LR).minimize(X)
    
    return OUT

result_dir = './CIFAR_RESULT_adagrad/' # directory where the results from the training are saved

ntrain     = 1000 # per class
ntest      = 100 # per class
nclass     = 10 # number of classes
imsize     = 28
nchannels  = 1
batchsize  = 256
Train      = np.zeros((ntrain*nclass,imsize,imsize,nchannels))
Test       = np.zeros((ntest*nclass,imsize,imsize,nchannels))
LTrain     = np.zeros((ntrain*nclass,nclass))
LTest      = np.zeros((ntest*nclass,nclass))
itrain     = -1
itest      = -1


for iclass in range(0, nclass):
    for isample in range(0, ntrain):
        path = '/Deep_learning/CIFAR10/Train/%d/Image%05d.png' % (iclass,isample)
        im = misc.imread(path); # 28 by 28
        im = im.astype(float)/255
        itrain += 1
        Train[itrain,:,:,0] = im
        LTrain[itrain,iclass] = 1 # 1-hot lable
    for isample in range(0, ntest):
        path = '/Deep_learning/CIFAR10/Test/%d/Image%05d.png' % (iclass,isample)
        im = misc.imread(path); # 28 by 28
        im = im.astype(float)/255
        itest += 1
        Test[itest,:,:,0] = im
        LTest[itest,iclass] = 1 # 1-hot lable



#for LR in [0.001]:
#    for Optimize in ["Momentum"]:
for iterate in range(1):
#    if iterate==0:
#        LR=0.1
#        momentum_val=0.9
#        Optimize="Momentum"
    if iterate==1:
        LR=0.1
        momentum_val=0
        Optimize="Adagrad"
    else:
        LR=0.001
        momentum_val=0
        Optimize="RMSprop"
    hparam = "Opt%s_LR_%s_mom%s" % (Optimize, LR, momentum_val)
    tf.reset_default_graph()   
    sess = tf.InteractiveSession()
    
    
    train_bit = tf.placeholder(tf.bool)
    cifar_data = tf.placeholder("float", shape=[None,imsize,imsize,nchannels]) #tf variable for the data, remember shape is [None, width, height, numberOfChannels]
    cifar_target = tf.placeholder("float", shape=[None,nclass]) #tf variable for labels
    print('Starting run for %s' % hparam)
    # --------------------------------------------------
    # model
    #create your model
    # Conv layer 1
    W_conv1 = weight_variable([5, 5, 1, 32])

    b_conv1 = bias_variable([32])
    
    h1_out=conv2d(cifar_data, W_conv1) + b_conv1
    h1_norm=batch_norm(h1_out, decay=0.9, updates_collections=None, is_training=train_bit)
    
    h_conv1 = tf.nn.relu(h1_norm)
    h_pool1 = max_pool_2x2(h_conv1)
    
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h2_out=conv2d(h_pool1, W_conv2) + b_conv2
    h2_norm=batch_norm(h2_out, decay=0.9, updates_collections=None, is_training=train_bit)
    
    h_conv2 = tf.nn.relu(h2_norm)
    h_pool2 = max_pool_2x2(h_conv2)
    
    W_fc1 = weight_variable([7*7*64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    
    h_fc1out=tf.matmul(h_pool2_flat, W_fc1) + b_fc1
    h_fc1norm= batch_norm(h_fc1out, decay=0.9,  updates_collections=None, is_training=train_bit)  
        
    h_fc1 = tf.nn.relu(h_fc1norm)
    keep_prob = tf.placeholder("float")
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    
    W_fc2 = weight_variable([1024, nclass])
    b_fc2 = bias_variable([nclass])
    
    forward = (tf.matmul(h_fc1_drop, W_fc2) + b_fc2)





    # --------------------------------------------------
    # loss
    #set up the loss, optimization, evaluation, and accuracy
    
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=forward, labels=cifar_target))

    optimizer= Optimisation(cross_entropy,Optimize,LR,momentum_val)                    
    evaluation = tf.equal(tf.argmax(forward,1), tf.argmax(cifar_target,1))
    accuracy = tf.reduce_mean(tf.cast(evaluation, "float"))
    
    
    
    # Add a scalar summary for the snapshot loss.
    acc_summary = tf.summary.scalar('Train_Accuracy', accuracy)# summary for accuracy
    loss_summary = tf.summary.scalar('Train_Loss', cross_entropy) # summary for loss 
    test_acc_summary = tf.summary.scalar('Test_Accuracy', accuracy)# summary for accuracy
    test_loss_summary = tf.summary.scalar('Test_Loss', cross_entropy) # summary for loss 
    init = tf.global_variables_initializer()
    
    
    # Build the summary operation based on the TF collection of Summaries.
    create_summaries(W_conv1, "Weight_conv1")
    create_summaries(h_conv1, "h_conv1")
    create_summaries(h_conv2, "h_conv2")
    grid = put_kernels_on_grid (W_conv1)
    weight_visualise= tf.summary.image('conv1/kernels', grid)
    summary_op = tf.summary.merge_all()
    
    # Create a saver for writing training checkpoints.
    #saver = tf.train.Saver()
    
    # Instantiate a SummaryWriter to output summaries and the Graph.
    summary_writer = tf.summary.FileWriter(result_dir+hparam, sess.graph)
    
    
    # --------------------------------------------------
    # optimization
    
    sess.run(tf.global_variables_initializer())
    batch_xs = np.zeros((batchsize,imsize,imsize,nchannels)) #setup as [batchsize, width, height, numberOfChannels] and use np.zeros()
    batch_ys = np.zeros((batchsize,nclass)) #setup as [batchsize, the how many classes]
    
    nsamples = nclass * ntrain
    max_step = 4000
    for i in range(max_step): # try a small iteration size once it works then continue
        perm = np.arange(nsamples)
        np.random.shuffle(perm)
        for j in range(batchsize):
            batch_xs[j,:,:,:] = Train[perm[j],:,:,:]
            batch_ys[j,:] = LTrain[perm[j],:]
        optimizer.run(feed_dict={cifar_data: batch_xs, cifar_target: batch_ys,train_bit: True, keep_prob: 0.5}) # dropout only during training
    
    
        if i%100 == 0:
            train_accuracy = accuracy.eval(feed_dict={cifar_data: batch_xs, cifar_target: batch_ys,train_bit: True, keep_prob: 1.0})
            print("step %d, training accuracy %g" % (i, train_accuracy))
            print("test accuracy %g"%accuracy.eval(feed_dict={cifar_data: Test, cifar_target: LTest,train_bit: False,keep_prob: 1.0}))
            #calculate train accuracy and print it
    
        # save the checkpoints every 10 iterations
        if i % 100 == 0 or i == max_step:
           
    	# Update the events file which is used to monitor the training (in this case,
    	# only the training loss is monitored)
            test_acc_sum,test_loss_sum = sess.run([test_acc_summary,test_loss_summary], feed_dict={cifar_data: Test, cifar_target: LTest ,train_bit: False,keep_prob: 1.0})                            
            summary_writer.add_summary(test_acc_sum, i)
            summary_writer.add_summary(test_loss_sum, i)
            
            train_acc_sum,train_loss_sum = sess.run([acc_summary,loss_summary], feed_dict={cifar_data: batch_xs, cifar_target: batch_ys,train_bit: True, keep_prob: 1.0})                       
            summary_writer.add_summary(train_acc_sum, i)
            summary_writer.add_summary(train_loss_sum, i)
            summary_str = sess.run(summary_op, feed_dict={cifar_data: Test, cifar_target: LTest, train_bit: False, keep_prob: 1.0})
            summary_writer.add_summary(summary_str, i)  
            
            summary_writer.flush()
            
   
    # --------------------------------------------------
    # test
    
    print("test accuracy %g"%accuracy.eval(feed_dict={cifar_data: Test, cifar_target: LTest,train_bit: False, keep_prob: 1.0}))
    
    
    
    sess.close()

