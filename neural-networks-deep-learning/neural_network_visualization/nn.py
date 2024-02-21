import tensorflow as tf
from tensorflow import keras
print(tf.__version__)


class nn(tf.keras.Model):

    def __init__(self): 
        super(nn, self).__init__()
        self.fc1 = tf.keras.layers.Dense(32*32, activation = "relu")
        self.fc2 = tf.keras.layers.Dense(32*32, activation = "relu")
        self.fc3 = tf.keras.layers.Dense(32*32, activation = "relu")
        self.fc4 = tf.keras.layers.Dense(10)
    
    def forward(self, inputs): 
        x = tf.reshape(inputs, [-1, 28*28])
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return tf.nn.log_softmax(x, axis=1)

model = nn()

