
"""
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                        %
%           Simple recurrent models in tensorflow (MNIST data)           %
%                                                                        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Summary: This script is used to create a RNN model in tensorflow for classifying MNIST


 Author: Akshay Sujatha Ravindran
 email: akshay dot s dot ravindran at gmail dot com
 Dec 23rd 2018
"""
import tensorflow as tf 
from tensorflow.python.ops import rnn
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True) #call mnist function


# Initialization
learningRate  = 0.001
trainingIters = 22000
batchSize     = 110
displayStep   = 10
nInput        = 28 #we want the input to take the 28 pixels
nSteps        = 28 #every 28
nHidden       = 512 #number of neurons for the RNN
nClasses      = 10 #this is MNIST so you know

x             = tf.placeholder('float', [None, nSteps, nInput])
y             = tf.placeholder('float', [None, nClasses])
weights       = {'out': tf.Variable(tf.random_normal([nHidden, nClasses]))}
biases        = {'out': tf.Variable(tf.random_normal([nClasses]))}

def RNN(x, weights, biases):
	x = tf.transpose(x, [1,0,2])
	x = tf.reshape(x, [-1, nInput])
#	x = tf.split(0, nSteps, x) #configuring so you can get it as needed for the 28 pixels

	#basicrnnCell = rnn_cell.BasicRNNCell(nHidden)
	#lstmCell = rnn_cell.BasicLSTMCell(nHidden, forget_bias=1.0)
	gruCell = tf.contrib.rnn.GRUCell(nHidden)


	outputs, states = rnn.raw_rnn(gruCell, x, dtype=tf.float32) #for the rnn where to get the output and hidden state

	return tf.matmul(outputs[-1], weights['out'])+ biases['out']



#optimization
#create the cost, optimization, evaluation, and accuracy

pred = RNN(x, weights, biases)
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learningRate).minimize(cost)
#optimizer = tf.train.RMSPropOptimizer(learning_rate=learningRate).minimize(cost)

correctPred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))
init = tf.initialize_all_variables()

with tf.Session() as sess:
	sess.run(init)
	step = 1

	testData = mnist.test.images.reshape((-1, nSteps, nInput))
	testLabel = mnist.test.labels

	while step* batchSize < trainingIters:
		batchX, batchY = mnist.train.next_batch(batchSize) #mnist has a way to get the next batch
		batchX = batchX.reshape((batchSize, nSteps, nInput))
		sess.run(optimizer, feed_dict={x: batchX, y: batchY})

		if step % displayStep == 0:
			acc = sess.run(accuracy, feed_dict={x: batchX, y: batchY})
			loss = sess.run(cost, feed_dict={x: batchX, y: batchY})
			testacc = sess.run(accuracy, feed_dict={x: testData, y: testLabel})
			print("Iter " + str(step*batchSize) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc) + ", Test Accuracy= " + "{:.5f}".format(testacc))
		step +=1
	print('Optimization finished')


	print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: testData, y: testLabel}))
