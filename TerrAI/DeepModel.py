import time
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models, optimizers
from tensorflow.keras.metrics import binary_accuracy, Recall, Precision, CategoricalAccuracy
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import *
from keras import backend as K

# See: https://datascience.stackexchange.com/questions/45165/how-to-get-accuracy-f1-precision-and-recall-for-a-keras-model
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

# See: https://datascience.stackexchange.com/questions/45165/how-to-get-accuracy-f1-precision-and-recall-for-a-keras-model
def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

# See: https://datascience.stackexchange.com/questions/45165/how-to-get-accuracy-f1-precision-and-recall-for-a-keras-model
def f1_score(y_true, y_pred):
    y_pred = tf.math.round(y_pred)
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

class DeepModel():
    def __init__(self, checkpoint_path, output_size, input_size):
        self.checkpoint_path = checkpoint_path
        self.input_size = input_size
        self.layer_index = 0
        self.output_size = int(output_size)
    
    def run(self):
        model = models.Sequential()
        model.add(Conv2D(input_shape=(9, 5, 5), filters=96, kernel_size=(3,3), padding="same"))
        model.add(Activation('relu'))
        model.add(Conv2D(filters=96, kernel_size=(3,3), strides=(1,1), padding="same"))
        model.add(Activation('relu'))
        #model.add(Dropout(0.2))
        model.add(Conv2D(filters=96, kernel_size=(3,3), strides=(1,1), padding="same"))
        model.add(Activation('relu'))
        model.add(Conv2D(filters=96, kernel_size=(3,3), strides=(1,1), padding="same"))
        model.add(Activation('relu'))
        #model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(BatchNormalization())
        model.add(Dense(256))
        model.add(Activation('relu'))
        model.add(Dense(self.output_size, activation="sigmoid"))
        model.compile(optimizer=optimizers.Adam(), loss=tf.keras.losses.BinaryCrossentropy(), metrics=[binary_accuracy, Recall(), Precision(), f1_score])
        return model
        
    """def run(self):
        inputs = keras.Input(shape=(self.input_size))
        
        w_2 = self.append_dense_layer(inputs, 'DENSE-1')
        w_2 = self.append_noise_layer(w_2, 'DENSE-1')
        w_2 = self.append_dense_layer(w_2, 'DENSE-1')
        w_2 = self.append_noise_layer(w_2, 'DENSE-1')
        w_o = self.append_dense_layer(w_2, 'DENSE-1')
        
        outputs = Dense(self.output_size, activation='sigmoid')(w_o)
    
        model = models.Model(inputs=inputs, outputs=outputs, name="DENSE")
        model.compile(optimizer=optimizers.Adam(), loss=tf.keras.losses.BinaryCrossentropy(), metrics=[binary_accuracy, Recall(), Precision(), f1_score])
        
        return model"""

    def append_dense_layer(self, x, prefix):
        self.layer_index += 1
        x = Dense(40, activation='relu', name=f"{prefix}-DENSE-{self.layer_index}")(x)
        x = BatchNormalization(name=f"{prefix}-NORM-{self.layer_index}")(x)
        x = Dropout(0.5, name=f"{prefix}-DROP-{self.layer_index}")(x)
        return x

    def append_gru_layer(self, x, prefix, return_sequences = True, first_layer = False):
        self.layer_index += 1
        gru_neuron_count = 512
        
        x = GRU(gru_neuron_count, return_sequences=return_sequences, recurrent_dropout=0, reset_after=True,recurrent_activation='sigmoid', name=f"{prefix}-GRU-{self.layer_index}")(x)
        
        return x
    
    def append_noise_layer(self, x, prefix):
        x = GaussianNoise(0.2)(x)
        
        return x
    
    def train(self, train_dataset, test_dataset, class_weight={0: 1.0, 1: 1}, epochs = 10, batch_size = 200):
        self.model = self.run()

        tensorboard = TensorBoard(log_dir="logs/{}".format(f"PRED-{int(time.time())}"), update_freq='batch')
        cp_callback = ModelCheckpoint(filepath=self.checkpoint_path,save_weights_only=True,verbose=0, save_freq='epoch')
        
        self.model.summary()
        
        self.model.fit(  
            train_dataset,
            validation_data=test_dataset,
            initial_epoch=0, 
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight,
            callbacks=[cp_callback, tensorboard],
            shuffle=True)
    
    def evaluate(self, dataset):
        self.model.evaluate(dataset)
    
    def prod(self):
        self.model = self.run()
        self.model.load_weights(self.checkpoint_path)
    
    def predict(self, dataset):
        return self.model.predict(dataset)