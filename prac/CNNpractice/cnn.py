from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(".", one_hot=true,reshape=False)

import tensorflow as tf

# Parameters
learning_rate = 0.00001
epochs = 10
batch_size = 128

# Number of samples to calculate validation and accuracy
# Decrease this if you're running out to memory to calculate accuracy

test_valid_size = 256

# Network Parameters
n_classes = 10 # MNIST total classes （0-9 digits）
dropout= 0,75 # Probability to keep units

# Store layers weight & bias
weights = {
    'wc1':tf.Variable(tf.random_normal([5,5,1,32])),
    'wc2':tf.Variable(tf.random_normal([5,5,32,64])),
    'wd1':tf.Variable(tf.random_normal([7*7*64,1024])),
    'out':tf.Variable(tf.random_normal([1024,n_classes]))}

bias = {
    'bc1':tf.Variable(tf.random_normal([32])),
    'bc2':tf.Variable(tf.random_normal([64])),
    'bd1':tf.Variable(tf.random_normal([1024])),
    'out':tf.Variable(tf.random_normal([n_classes]))}

# if Graph input
x = tf.placeholder(tf.float32,[None,28,28,1])
y = tf.placeholder(tf.float32,[None,n_classes])
keep_prob = tf.placeholder(tf.float32)

# Model
logits = conv_net(x,weights,bias,keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits,y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)\
    .minimize(cost)

# Accuracy
correct_pred = tf.equal(tf.argmax(logits,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the Graph
with tf.Session() as sess:
    sess.run(init)

    for epoch in range(epochs):
        for batch in range(mnist.train.num_examples//batch_size):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            sess.run(optimizer, feed_dict={
                x:batch_x,
                y:batch_y,
                keep_prob:dropout})

                #Calculate batch loss and accuracy
                loss = sess.run(cost,feed_dict={
                    x:batch_x,
                    y:batch_y,
                    keep_prob:1.})

                valid_acc = sess.run(accuracy, feed_dict={
                    x:mnist.validation.images[:test_valid_size],
                    y:mnist.validation.labels[:test_valid_size],
                    keep_prob:1.})

                print('Epoch{:>2}, Batch{:>3} -'
                      'Loss :{:>10.4f} Validation Accuracy: {:.6f}'.format(
                      epoch + 1,
                      batch + 1,
                      loss,
                      valid_acc))

# Calculate the Test Accuracy
test_acc = sess.run(accuracy,feed_dict={
    x:mnist.test.images[:test_valid_size],
    y:mnist.test.labels[:test_valid_size],
    keep_prob:1.})
print('TEsting Accuracy:{}'.format(test_acc))
