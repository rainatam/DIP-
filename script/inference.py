with tf.Session() as sess:

    data = np.load('train_image.npy')

    _, _, HEIGHT, WIDTH, CHANNEL = data.shape
    
    X_base = tf.placeholder(tf.float32, [None, HEIGHT, WIDTH, CHANNEL])
    X_inf = tf.placeholder(tf.float32, [None, HEIGHT, WIDTH, CHANNEL])

    sim = Distinguisher(X_base, X_pos, sess).sim

    inf = load_image()

    res = sess.run([sim], feed_dict={X_base: , X_inf: })

    for i in range(50):
        for j in range(10):
            
            