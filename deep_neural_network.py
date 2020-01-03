import tensorflow as tf
import visualizations as vs
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)
from sklearn.model_selection import KFold
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

def deep_neural_network():
    # Definition parameters
    NUM_CLASSES = 10  # number of  output classes
    NUM_FEATURES = 784  # number of data features

    LEARNING_RATE = 0.004
    NUM_ITERS = 5000
    NUM_ITERS_2 = 3000
    DISPLAY_STEP = 100
    BATCH_SIZE = 128
    tf.set_random_seed(0)

    NUM_NEUR_1 = 256  # Number of neurons in first layer
    NUM_NEUR_2 = 128  # Number of neurons in second layer
    NUM_NEUR_3 = 64  # Number of neurons in third layer
    NUM_NEUR_4 = 32  # Number of neurons in third layer

    C1 = 4
    C2 = 8
    C3 = 16
    FC4 = 256
    Filter_size_1 = 4
    Filter_size_2 = 4
    Filter_size_3 = 3

    # Import MNIST data
    mnist = read_data_sets("MNISTdata", one_hot=True, reshape=False, validation_size=0)

    # create tf Graph input
    X = tf.placeholder(tf.float32, [None, 28, 28, 1])
    Y_ = tf.placeholder(tf.float32, [None, 10])
    pkeep = tf.placeholder(tf.float32)

    # Flatten images (unrole each image row by row)
    XX = tf.reshape(X, [-1, 784])

    # Store layers weight and biases
    W1 = tf.Variable(tf.truncated_normal([Filter_size_1, Filter_size_1, 1, C1], stddev=0.1))
    b1 = tf.Variable(tf.truncated_normal([C1], stddev=0.1))

    W2 = tf.Variable(tf.truncated_normal([Filter_size_2, Filter_size_2, C1, C2], stddev=0.1))
    b2 = tf.Variable(tf.truncated_normal([C2], stddev=0.1))

    W3 = tf.Variable(tf.truncated_normal([Filter_size_3, Filter_size_3, C2, C3], stddev=0.1))
    b3 = tf.Variable(tf.truncated_normal([C3], stddev=0.1))

    #fully connected layer
    W4 = tf.Variable(tf.truncated_normal([7*7*C3, FC4], stddev=0.1))
    b4 = tf.Variable(tf.truncated_normal([FC4], stddev=0.1))

    W5 = tf.Variable(tf.truncated_normal([FC4, 10], stddev=0.1))
    b5 = tf.Variable(tf.truncated_normal([10], stddev=0.1))

    # Create architecture
    # Create model
    stride = 1
    k = 2
    #first convolution layer
    Y1 = tf.nn.relu(tf.nn.conv2d(X, W1, strides=[1, stride, stride, 1], padding='SAME')+ b1)
    #second convolution layer with max_pooling
    Y2 = tf.nn.relu(tf.nn.conv2d(Y1, W2, strides=[1, stride, stride, 1], padding='SAME')+ b2)
    Y2 = tf.nn.max_pool(Y2, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')
    #third convolution layer with max pooling
    Y3 = tf.nn.relu(tf.nn.conv2d(Y2, W3, strides=[1, stride, stride, 1], padding='SAME'))
    Y3 = tf.nn.max_pool(Y3, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')
    # reshape the output from the third convolution for the fully connected layer
    Y3_ = tf.reshape(Y3, shape=[-1, 7*7*C3])
    #fourth fully connected layer
    Y4 = tf.nn.relu(tf.matmul(Y3_, W4) + b4)
    Logits = tf.matmul(Y4, W5) + b5
    Y = tf.nn.softmax(Logits)

    # Define loss
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Logits, labels=Y_)
    cross_entropy = tf.reduce_mean(cross_entropy) * 100

    # Define optimizer
    train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)

    # Evaluate model
    correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Matplotlib visualization
    allweights = tf.concat([tf.reshape(W1, [-1]), tf.reshape(W2, [-1]), tf.reshape(W3, [-1]), tf.reshape(W4, [-1]), tf.reshape(W5, [-1])], 0)
    allbiases = tf.concat([tf.reshape(b1, [-1]), tf.reshape(b2, [-1]), tf.reshape(b3, [-1]), tf.reshape(b4, [-1]), tf.reshape(b5, [-1])], 0)

    # Initializing the variables
    init = tf.global_variables_initializer()

    train_losses = list()
    train_acc = list()
    test_losses = list()
    test_acc = list()

    with tf.Session() as sess:
        # 5-fold cross_validation
        results = [];
        loss = [];
        kf = KFold(n_splits=5)
        for train_idx, val_idx in kf.split(mnist.train.images, mnist.train.labels):
            train_x = mnist.train.images[train_idx]
            train_y = mnist.train.labels[train_idx]
            val_x = mnist.train.images[val_idx]
            val_y = mnist.train.labels[val_idx]
            sess.run(init)
            for epoch in range(5):
                total_batch = int(train_x.shape[0] / 256)
                for i in range(total_batch):
                    batch_X = train_x[i * 256:(i + 1) * 256]
                    batch_Y = train_y[i * 256:(i + 1) * 256]
                    if i % DISPLAY_STEP == 0:
                        # Calculate training values for visualization
                        acc_trn, loss_trn, w, b = sess.run([accuracy, cross_entropy, allweights, allbiases],
                                                           feed_dict={X: batch_X, Y_: batch_Y, pkeep: 1.0})
                        print("Epoch {} Trn acc={} , Trn loss={}".format(epoch, acc_trn, loss_trn))

                    # Run optimalization op (backpropagation)
                    sess.run(train_step, feed_dict={X: batch_X, Y_: batch_Y, pkeep: 0.75})
            acc_val, loss_val = sess.run([accuracy, cross_entropy], feed_dict={X: val_x, Y_: val_y, pkeep: 1.0})
            results.append(acc_val);
            loss.append(loss_val);
        print("Cross validation accuracy: {:.4f}".format(results[-1]))
        print("Cross validation loss: {:.4f}".format(loss[-1]))

    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)
        for i in range(NUM_ITERS + 1):
            batch_X, batch_Y = mnist.train.next_batch(BATCH_SIZE)
            if i % DISPLAY_STEP == 0:
                # Calculate training values for visualization
                acc_trn, loss_trn, w, b = sess.run([accuracy, cross_entropy, allweights, allbiases],
                                                   feed_dict={X: batch_X, Y_: batch_Y, pkeep: 1.0})

                acc_tst, loss_tst = sess.run([accuracy, cross_entropy],
                                             feed_dict={X: mnist.test.images, Y_: mnist.test.labels, pkeep: 1.0})

                print("#{} Trn acc={} , Trn loss={}, Tst acc={}, Tst loss = {}".format(i, acc_trn, loss_trn, acc_tst,
                                                                                       loss_tst))

                train_losses.append(loss_trn)
                train_acc.append(acc_trn)
                test_losses.append(loss_tst)
                test_acc.append(acc_tst)

                # Run optimalization op (backpropagation)
            sess.run(train_step, feed_dict={X: batch_X, Y_: batch_Y, pkeep: 0.75})

    # Display plot
    title = "Mnist_5_layer_conv"
    vs.my_plot(train_losses, train_acc, test_losses, test_acc, title, DISPLAY_STEP)