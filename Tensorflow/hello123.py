import tensorflow as tf

hello_constant = tf.constant('Hello World!')
with tf.Session() as sess:
    # Run the tf.constant operation in the session
    output = sess.run(hello_constant)
    print(output)
    
def run():
    output = None
    x = tf.placeholder(tf.int32)

    with tf.Session() as sess:
        output = sess.run(x,feed_dict={x:123})

    return output

print (run())


a = tf.add(5,2)
b = tf.sub(10,4)
c = tf.mul(2,5)
d = tf.div(10,5)


def math():
    with tf.Session() as sess:
        output1 = sess.run(a)
        output2 = sess.run(b)
        output3 = sess.run(c)
        output4 = sess.run(d)
        print(output1,output2,output3,output4)

math()
