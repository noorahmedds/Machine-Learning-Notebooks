import tensorflow as tf

# x1 = tf.constant(5)
# x2 = tf.constant(6)
#
#
# result = tf.scalar_mul(x1, x2)
#
# print (result)
#
# # sess = tf.Session()
# # print(sess.run(result))
# # sess.close()
#
# with tf.Session() as sess:
#     answer = sess.run(result)
#     print ("Answer: ", answer)

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

n_nodes_hidden1 = 1000
# n_nodes_hidden2 = 500
# n_nodes_hidden3 = 500

n_class = 10

batch_size = 100


x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

def neural_network_model(data):
    # now we create our hidden layer variables
    # our weights are always S(j+1) x S(j)
    hidden_layer1 = {'weights':tf.Variable(tf.random_normal([784, n_nodes_hidden1])), 'biases':tf.Variable(tf.random_normal([n_nodes_hidden1]))}

    output_layer = {'weights':  tf.Variable(tf.random_normal([n_nodes_hidden1, n_class])),
                    'biases' :  tf.Variable(tf.random_normal([n_class]))}

    z1 = tf.add(tf.matmul(data, hidden_layer1['weights']), hidden_layer1['biases'])
    # rectified lienear function which is the like the activation function
    # so we apply the the sigmoid function on our z values
    output_layer_x = tf.nn.relu(z1)

    output = tf.add(tf.matmul(output_layer_x, output_layer['weights']), output_layer['biases'])
    # fin_output = tf.nn.relu(output)

    return output

def train_neural_network(x):
    prediction = neural_network_model(x)

    # this is basically our cost function so we want to reduce our cost after this. This makes alot of sense
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits( logits = prediction, labels = y ) )
                                                                    # ^        ^ so you can see these two are like getting subtracted to check for the minimization
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    n_epochs = 10

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for epoch in range(0, n_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                # create a mini_batch
                epochx, epochy = mnist.train.next_batch(batch_size)

                # here we just calculate the epoch loss for this mini_batch
                _, c = sess.run([optimizer, cost], feed_dict = {x:epochx, y:epochy})
                epoch_loss += c
            print('Epoch:', epoch, "completed out of", n_epochs, "loss:", epoch_loss)


        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        # here were just checking the accuracy of our model with some test data in the mnist dataset
        print('Accuracy: ', accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))




train_neural_network(x);
