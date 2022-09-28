import tensorflow as tf
# tf.config.run_functions_eagerly(True)
# tf.compat.v1.disable_eager_execution()
# tf.config.run_functions_eagerly(True)
@tf.function
def inner_function(x, y, b):
  print(x.shape);
  print(y.shape);
  x = tf.matmul(x, y)
  x = x + b
  return x

# @tf.function
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
    W=tf.random.truncated_normal(shape=shape,mean=0,stddev=0.2);
    return W;
# Use the decorator to make `outer_function` a `Function`.
@tf.function
def outer_function(x):
  y = tf.constant([[2.0], [3.0]])
  b = tf.constant(4.0)

  return weight_variable(x)
#   return inner_function(x, y, b)

# Note that the callable will create a graph that
# includes `inner_function` as well as `outer_function`.
# print(outer_function(tf.constant([[1.0, 2.0]])).numpy())
# print(outer_function(tf.constant([[1.0, 2.0]])).numpy())
x=weight_variable([2,3]);
print(x)
print(tf.reduce_sum(x,axis=1,keepdims=True))
print(x/tf.reduce_sum(x,axis=1,keepdims=True))
# print(outer_function(tf.constant([1])))
# print(outer_function(tf.constant([1])))