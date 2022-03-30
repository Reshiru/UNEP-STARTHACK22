import time
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models, optimizers
from tensorflow.keras.metrics import binary_accuracy, Recall, Precision, CategoricalAccuracy
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import *
from keras import backend as K
from tensorflow.keras import regularizers

# Hyper parameter search
from kerastuner.tuners import BayesianOptimization #pip install -U keras-tuner
from kerastuner import HyperModel, Objective

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

class DeepModel(HyperModel):
    def __init__(self, checkpoint_path, output_size, input_size):
        self.checkpoint_path = checkpoint_path
        self.input_size = input_size
        self.layer_index = 0
        self.output_size = int(output_size)
    
    def run(self):
        model = models.Sequential()
        model.add(Conv2D(input_shape=(9, 5, 5), filters=64, kernel_size=(3,3), padding="same"))
        model.add(Activation('relu'))
        model.add(Dropout(0.1))
        model.add(Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding="same"))
        model.add(Activation('relu'))
        model.add(Dropout(0.1))
        model.add(Flatten())
        model.add(Dense(16))
        model.add(Activation('relu'))
        model.add(Dropout(0.1))
        model.add(Dense(self.output_size, activation="sigmoid"))
        model.compile(optimizer=optimizers.Adam(), loss=tf.keras.losses.BinaryCrossentropy(), metrics=[binary_accuracy, Recall(), Precision(), f1_score])
        return model
    
    def build(self, hp):
        model = models.Sequential()
        model.add(Conv2D(input_shape=(9, 5, 5), filters=hp.Int('f_1',min_value=4,max_value=64,step=4,default=32), kernel_size=(3,3), padding="same"))
        model.add(Activation('relu'))
        model.add(Dropout(hp.Float('d_1',min_value=0.1,max_value=0.5,step=0.1,default=0.2)))
        model.add(Conv2D(filters=hp.Int('f_2',min_value=4,max_value=64,step=2,default=8), kernel_size=(3,3), strides=(1,1), padding="same"))
        model.add(Activation('relu'))
        model.add(Dropout(hp.Float('d_2',min_value=0.1,max_value=0.5,step=0.1,default=0.2)))
        model.add(Flatten())
        model.add(Dense(hp.Int('f_3',min_value=4,max_value=64,step=4,default=16)))
        model.add(Activation('relu'))
        model.add(Dropout(hp.Float('d_3',min_value=0.1,max_value=0.5,step=0.1,default=0.5)))
        model.add(Dense(self.output_size, activation="sigmoid"))
        model.compile(optimizer=optimizers.Adam(), loss=tf.keras.losses.BinaryCrossentropy(), metrics=[binary_accuracy, f1_score])
        return model
        
    def search_better_model(self, train_dataset, test_dataset, class_weight, batch_size):
        search_opt = BayesianOptimization(hypermodel=self, objective=Objective("val_f1_score", direction="max"), max_trials=256, executions_per_trial=1, overwrite=False)
        search_opt.search(train_dataset, validation_data=test_dataset, epochs=20, batch_size=batch_size, class_weight=class_weight, verbose=0)
        best_model = search_opt.get_best_models(num_models=1)[0]
        print(search_opt.get_best_hyperparameters(num_trials=1)[0].values)
        best_model.summary()
        return search_opt
        
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