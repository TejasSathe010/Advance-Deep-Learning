# Import Libraries
import tensorflow as tf
import numpy as np
from tensorflow import keras
import os

# Define the Regressor Class
class Regressor(keras.layers.Layer):

    def __init__(self):
        super(Regressor, self).__init__()
        # here must specify shape instead of tensor!
        # [dim_in, dim_out]
        self.w = self.add_variable('Variables', [13, 1])
        # [dim_out]
        self.b = self.add_variable('Bias', [1])

        print(self.w.shape, self.b.shape)
        print(type(self.w), tf.is_tensor(self.w), self.w.name)
        print(type(self.b), tf.is_tensor(self.b), self.b.name)


    def call(self, x):
        x = tf.matmul(x, self.w) + self.b
        return x

def main():
    tf.random.set_seed(22)
    np.random.seed(22)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    assert tf.__version__.startswith('2.')

    # Load the Housing regression data
    (x_train, y_train), (x_val, y_val) = keras.datasets.boston_housing.load_data()
    # Preprocess the Data
    x_train, x_val = x_train.astype(np.float32), x_val.astype(np.float32)
    # (404, 13) (404,) (102, 13) (102,)
    print(x_train.shape, y_train.shape, x_val.shape, y_val.shape)
    
    # Covert the Numpy data to tf.Data
    db_train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(64)
    db_val = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(102)


    # Define the Model
    model = Regressor()
    # Define the Loss Function
    loss_fn = keras.losses.MeanSquaredError()
    # Define the Optimizer
    optimizer = keras.optimizers.Adam(learning_rate=1e-2)

    for epoch in range(200):
      for step, (x, y) in enumerate(db_train):

            with tf.GradientTape() as tape:
                # [b, 1]
                logits = model(x)
                # [b]
                logits = tf.squeeze(logits, axis=1)
                # [b] vs [b]
                loss = loss_fn(y, logits)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

      print(epoch, 'loss:', loss.numpy())


      if epoch % 10 == 0:
          for x, y in db_val:
              # [b, 1]
              logits = model(x)
              # [b]
              logits = tf.squeeze(logits, axis=1)
              # [b] vs [b]
              loss = loss_fn(y, logits)

              print(epoch, 'val loss:', loss.numpy())


if __name__ == '__main__':
    main()
