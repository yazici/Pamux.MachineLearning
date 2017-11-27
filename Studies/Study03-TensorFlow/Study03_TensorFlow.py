
import input_data
mnist = input_data.read_data_sets("D:\\Workspace\\EngTools\\Temp\\Data", one_hot=True)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # AVX instruction warning off
import tensorflow as tf

hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
sess.run(hello)
a = tf.constant(10)
b = tf.constant(32)
sess.run(a + b)
42
sess.close()

# Hyperparameters (tuning knobs)
learning_rate = 0.01
training_iteration = 30
batch_size = 100
display_step = 2


x = tf.placeholder("float", [None, 784]) # mnist data image of shape 28*28 = 784
y = tf.placeholder("float", [None, 10]) # Digits 0-9 (10 classes)

# Create a model
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

with tf.name_scope("Wx_b") as scope:
    # Construct a linear model
    model = tf.nn.softmax(tf.matmul(x, W) + b)

w_hist = tf.summary.histogram("weights", W)
b_hist = tf.summary.histogram("biases", b)

with tf.name_scope("cost_function") as scope:
    # minimize error using cross entropy
    cost_function = tf.reduce_sum(y*tf.log(model))
    tf.summary.scalar("cost_function", cost_function)

with tf.name_scope("train") as scope:
    optimizer  = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

initialize  = tf.global_variables_initializer()

#merged_train = tf.summary.merge([w_hist, b_hist]) 
merged_summary_op  = tf.summary.merge_all()

#merged_summary_op  = tf.summary.merge([w_hist, b_hist]) 

with tf.Session() as sess:
    sess.run(initialize)

    summary_writer = tf.summary.FileWriter('D:\\Workspace\\EngTools\\Temp\\Logs', graph = sess.graph)

    # Training cycle

    for iteration in range(training_iteration):
        average_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(optimizer, feed_dict = {x: batch_xs, y: batch_ys})
            average_cost += sess.run(cost_function, feed_dict = {x: batch_xs, y: batch_ys}) / total_batch
            summary_str = sess.run(merged_summary_op, feed_dict = {x: batch_xs, y: batch_ys})
            summary_writer.add_summary(summary_str, global_step = total_batch+i)
        if iteration % display_step == 0:
            print ("Iteration: ", '%0d' % (iteration + 1), "cost=", "{:.9f}".format(average_cost))

    print ("Training completed")

    predictions = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))

    accuracy = tf.reduce_mean(tf.cast(predictions, "float"))

    print ("accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))