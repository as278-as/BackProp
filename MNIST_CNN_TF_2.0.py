
from audioop import cross
import os
import time
import numpy as np
import datetime
# Load MNIST dataset
#EAGER MODE EXECUTION
# Import Tensorflow and start a session
import tensorflow as tf
# sess = tf.InteractiveSession()
# sess=tf.compat.v1.InteractiveSession()
# tf.compat.v1.disable_eager_execution()
import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

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
    W=tf.random.truncated_normal(shape,stddev=0.1);
    return tf.Variable(W);


def bias_variable(shape):
    '''
    Initialize biases
    :param shape: shape of biases, e.g. [Cout] where
    Cout: the number of filters
    :return: a tensor variable for biases with initial values
    '''

    # IMPLEMENT YOUR BIAS_VARIABLE HERE
    b=tf.constant(0.1,shape=shape);
    return tf.Variable(b);


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
    h_conv=tf.nn.conv2d(x,W,strides=[1,1,1,1],padding="SAME");
    return h_conv;

def max_pool_2x2(x):
    '''
    Perform non-overlapping 2-D maxpooling on 2x2 regions in the input data
    :param x: input data
    :return: the results of maxpooling (max-marginalized + downsampling)
    '''

    # IMPLEMENT YOUR MAX_POOL_2X2 HERE

    h_max=tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME");
    
    return h_max


    
def main():
    # Specify training parameters
    result_dir = './results/' # directory where the results from the training are saved
    max_step = 5500 # the maximum iterations. After max_step iterations, the training will stop no matter what

    start_time = time.time() # start timing

    # FILL IN THE CODE BELOW TO BUILD YOUR NETWORK
    W_conv1 = weight_variable([5,5,1,32]);
    b_conv1 = weight_variable([32]);
  
    W_conv2 = weight_variable([5,5,32,64]);
    b_conv2 = weight_variable([64]);

    W_fc1 = weight_variable([7 * 7 * 64, 1024]);
    b_fc1 =weight_variable([1024]);

    W_fc2 =weight_variable([1024, 10]);
    b_fc2 =bias_variable([10]);
    
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    loss_dir = 'logs/gradient_tape/'
    loss_summary_writer = tf.summary.create_file_writer(loss_dir)
    train_step = tf.keras.optimizers.Adam(learning_rate=0.005);
    for i in range(max_step):
        batch = mnist.train.next_batch(50) # make the data batch, which is used in the training iteration.
                                            # the batch size is 50

        x=batch[0];

        y=batch[1]; 

        x_image = tf.reshape(x, [-1, 28, 28, 1]);
        
        with tf.GradientTape() as tape:
            # tape.watch(W_conv1);
            h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1);
            h_pool1 = max_pool_2x2(h_conv1);
        
            # second convolutional layer
            
            h_conv2 =tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2);
            h_pool2 =max_pool_2x2(h_conv2);
        
            # densely connected layer
            
            h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64]);
            h_fc1 =tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1);
        
            # dropout
            h_fc1_drop =tf.nn.dropout(h_fc1, 0.2);
            # print(h_fc1_drop);
            # softmax
           
            temp=tf.matmul(h_fc1_drop, W_fc2) + b_fc2;
            print(temp)
            temp=temp /tf.reduce_sum(temp,axis=1,keepdims=True)
            print(temp);
            # # 
            y_conv =tf.nn.softmax(temp, name='y_conv');
   
            cross_entropy =tf.reduce_mean(-tf.reduce_sum(y* tf.math.log(y_conv)));
            


       
        grads=tape.gradient(cross_entropy,[W_conv1,W_conv2,b_conv1,b_conv2,W_fc1,W_fc2,b_fc1,b_fc2]);
        # print(grads)
        gradients = [(tf.clip_by_value(grad, -1.0, 1.0)) for grad in grads]

        train_step.apply_gradients(zip(gradients,[W_conv1,W_conv2,b_conv1,b_conv2,W_fc1,W_fc2,b_fc1,b_fc2]));
        # print(train_step.weights)
        correct_prediction =tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1));
        accuracy =tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy');
        

        if i%100 == 0:
            print("test accuracy %g"%(accuracy));
            # print(y_conv);

        # save the checkpoints every 1100 iterations
        if i % 100 == 0 or i == max_step:
            with loss_summary_writer.as_default():
                tf.summary.scalar('loss', cross_entropy, step=i)

    stop_time = time.time()
    print('The training takes %f second to finish'%(stop_time - start_time))


   
if __name__ == "__main__":
    main()
