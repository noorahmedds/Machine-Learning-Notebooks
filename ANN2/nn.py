import tensorflow as tf
from create_sentiment_feature_set import create_feature_sets_and_labels

train_features, train_labels, text_features, text_labels = create_feature_sets_and_labels('pos.txt', 'neg.txt', 0.6)

n_nodes_hidden1 = 1000
n_nodes_hidden2 = 500
# n_nodes_hidden3 = 500

n_class = 2

batch_size = 100


x = tf.placeholder('float', [None, train_features[0]])
y = tf.placeholder('float')

def neural_network_model(data):
    # now we create our hidden layer variables
    # our weights are always S(j+1) x S(j)

    # a little miniscule detail. weights gets passed this list with S(j) x S(j+1) rather than S(j+1) x S(j) which in unituitive but makes kinda sense
    hidden_layer1 = {'weights':tf.Variable(tf.random_normal([train_features[0], n_nodes_hidden1])), 'biases':tf.Variable(tf.random_normal([n_nodes_hidden1]))}

    hidden_layer2 = {'weights':tf.Variable(tf.random_normal([n_nodes_hidden1, n_nodes_hidden2])), 'biases':tf.Variable(tf.random_normal([n_nodes_hidden2]))}

    output_layer = {'weights':  tf.Variable(tf.random_normal([n_nodes_hidden2, n_class])),
                    'biases' :  tf.Variable(tf.random_normal([n_class]))}


    # Basically the code below is forward prop
    # i.e. z = theta.x + b
    z1 = tf.add(tf.matmul(data, hidden_layer1['weights']), hidden_layer1['biases'])
    z2 = tf.add(tf.matmul(z1  , hidden_layer2['weights']), hidden_layer2['biases'])

    # rectified lienear function which is the like the activation function
    # so we apply the the sigmoid function on our z values
    output_layer_x = tf.nn.relu(z2)

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

            i = 0
            while i < len(train_x):
                start = i
                end = i+batch_size
                batch_x = np.array(train_features[start:end])
                batch_y = np.array(train_labels[start:end])

                # here we just calculate the epoch loss for this mini_batch
                _, c = sess.run([optimizer, cost], feed_dict = {x:epochx, y:epochy})
                epoch_loss += c

                i += batch_size
            print('Epoch:', epoch, "completed out of", n_epochs, "loss:", epoch_loss)


        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        # here were just checking the accuracy of our model with some test data in the mnist dataset
        print('Accuracy: ', accuracy.eval({x:test_features, y:test_labels}))




train_neural_network(x);
