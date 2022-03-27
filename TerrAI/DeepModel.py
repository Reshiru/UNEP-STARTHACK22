import time
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models, optimizers
from tensorflow.keras.metrics import binary_accuracy, Recall, Precision, CategoricalAccuracy
from tensorflow_addons.metrics import F1Score
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import *
#from keras import backend as K

class DeepModel():
    def __init__(self, checkpoint_path, output_size, input_size):
        self.checkpoint_path = checkpoint_path
        self.input_size = input_size
        self.layer_index = 0
        self.output_size = int(output_size)
        
    #def custom_activation(self, x):
    #    return tf.math.round(K.sigmoid(x))
    
    def run(self):
        inputs = keras.Input(shape=(self.input_size))
        
        w_2 = self.append_dense_layer(inputs, 'DENSE-1')
        w_2 = self.append_dense_layer(w_2, 'DENSE-1')
        w_o = self.append_dense_layer(w_2, 'DENSE-1')
        
        outputs = Dense(self.output_size, activation='sigmoid')(w_o)
    
        model = models.Model(inputs=inputs, outputs=outputs, name="DENSE")
        model.compile(optimizer=optimizers.Adam(), loss=tf.keras.losses.BinaryCrossentropy(), metrics=[binary_accuracy, Recall(), Precision(), F1Score(num_classes=2, average='micro')])
        
        return model

    def append_dense_layer(self, x, prefix):
        self.layer_index += 1
        x = Dense(64, activation='relu', name=f"{prefix}-DENSE-{self.layer_index}")(x)
        x = BatchNormalization(name=f"{prefix}-NORM-{self.layer_index}")(x)
        x = Dropout(0.2, name=f"{prefix}-DROP-{self.layer_index}")(x)
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