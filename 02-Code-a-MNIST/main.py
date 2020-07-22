# Import the Libraries
import tensorflow as tf
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics

# Download the Dataset
(xs, ys),_ = datasets.mnist.load_data()
print('datasets:', xs.shape, ys.shape, xs.min(), xs.max())

# Process the Dataset
xs = tf.convert_to_tensor(xs, dtype=tf.float32) / 255.
db = tf.data.Dataset.from_tensor_slices((xs,ys))
db = db.batch(32).repeat(10)

# Define the Model
network = Sequential([layers.Dense(256, activation='relu'),
                     layers.Dense(256, activation='relu'),
                     layers.Dense(256, activation='relu'),
                    layers.Dense(10)])

# Build the Model to Get the Parameters
network.build(input_shape=(None, 28*28))
# Take a look at the Summary of the Model
network.summary()

# Define the Optimizer Object
optimizer = optimizers.SGD(lr=0.01)
# Define the Accuracy Object
acc_meter = metrics.Accuracy()

# Training
for step, (x,y) in enumerate(db):

    with tf.GradientTape() as tape:
        # [b, 28, 28] => [b, 784]
        x = tf.reshape(x, (-1, 28*28))
        # [b, 784] => [b, 10]
        out = network(x)
        # [b] => [b, 10]
        y_onehot = tf.one_hot(y, depth=10)
        # [b, 10]
        loss = tf.square(out-y_onehot)
        # [b]
        loss = tf.reduce_sum(loss) / 32

    # Update the State of Accuracy Object
    acc_meter.update_state(tf.argmax(out, axis=1), y)
    
    # Compute the Gradients of Parameters wrt Loss Function 
    grads = tape.gradient(loss, network.trainable_variables)
    # Optimize the values of Parameters with the help of Grads
    optimizer.apply_gradients(zip(grads, network.trainable_variables))

    # Print the resulf after each 200 Steps
    if step % 200==0:

        print(step, 'loss:', float(loss), 'acc:', acc_meter.result().numpy())
        acc_meter.reset_states()

print('Final Accuracy of the Model: ', acc_meter.result().numpy())
