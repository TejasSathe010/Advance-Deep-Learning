import tensorflow as tf
from tensorflow import  keras
from tensorflow.keras import datasets, layers, optimizers, models
from tensorflow.keras import regularizers

# Define VGG16 Architecture Class
class VGG16(models.Model):

    def __init__(self, input_shape):
        """

        :param input_shape: [32, 32, 3]
        """
        super(VGG16, self).__init__()

        weight_decay = 0.000
        self.num_classes = 10

        # Define Building Block using Sequential API
        model = models.Sequential()



        # Block1
        # Conv1
        model.add(layers.Conv2D(64, (3, 3), padding='same',
                         input_shape=input_shape, kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.3))
        
        # Conv2
        model.add(layers.Conv2D(64, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization())
        
        # MaxPool
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))



        # Block2
        # Conv1
        model.add(layers.Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.4))

        # Conv2
        model.add(layers.Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization())

        # MaxPool
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))



        # Block3
        # Conv1
        model.add(layers.Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.4))

        # Conv2
        model.add(layers.Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.4))

        # Conv3
        model.add(layers.Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization())

        # MaxPool
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))



        # Block4
        # Conv1
        model.add(layers.Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.4))

        # Conv2
        model.add(layers.Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.4))

        # Conv3
        model.add(layers.Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization())

        # MaxPool
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))


        # Block5
        # Conv1
        model.add(layers.Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.4))

        # Conv2
        model.add(layers.Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.4))

        # Conv3
        model.add(layers.Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization())

        # MaxPool
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Dropout(0.5))

        
        # Flatten
        model.add(layers.Flatten())
        # Dense
        model.add(layers.Dense(512,kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization())

        # Dropout
        model.add(layers.Dropout(0.5))
        # Output Layer
        model.add(layers.Dense(self.num_classes))
        # model.add(layers.Activation('softmax'))

        # Define the Model
        self.model = model


    def call(self, x):
        x = self.model(x)
        return x
