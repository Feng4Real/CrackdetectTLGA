#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import preprocess_input
import pickle
from sklearn.metrics import classification_report,confusion_matrix

# Define the custom Spatial Attention Layer
class SpatialAttention(layers.Layer):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = layers.Conv2D(filters=1, kernel_size=kernel_size, padding='same', activation='sigmoid')

    def call(self, inputs):
        avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)
        concat = tf.concat([avg_pool, max_pool], axis=-1)
        attention = self.conv(concat)
        return inputs * attention

# Data preparation
train_data_path = "../crack_images"
test_data_path = "../crack_test"
train_datagen = ImageDataGenerator(validation_split=0.3, preprocessing_function=preprocess_input)
train_generator = train_datagen.flow_from_directory(train_data_path, target_size=(224, 224), batch_size=64, shuffle=True, class_mode='categorical', subset='training')
validation_generator = train_datagen.flow_from_directory(train_data_path, target_size=(224, 224), batch_size=64, class_mode='categorical', subset='validation')
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_generator = test_datagen.flow_from_directory(
    test_data_path,  # Specify the path to your test data
    target_size=(224, 224),
    batch_size=64,
    shuffle=False,  # No shuffling for the test set
    class_mode='categorical'
)


# Base model ResNet50
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  
x = base_model.output
x = SpatialAttention()(x)
x = layers.GlobalAveragePooling2D()(x)  

# Define function to create and train model based on GA parameters
def create_model(attention_output, num_layers, neurons_per_layer):
    x = attention_output
    for _ in range(num_layers):
        x = layers.Dense(neurons_per_layer[_], activation='relu')(x)
        x = layers.Dropout(0.5)(x)
    output_layer = layers.Dense(2, activation='softmax')(x) 
    model = Model(inputs=base_model.input, outputs=output_layer)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = create_model(x, 4, [66, 805, 218, 382])

model.summary()
history = model.fit(
        train_generator,
        epochs=15,
        validation_data=validation_generator,
        verbose=1
    )

with open('training_history.pkl', 'wb') as file:
    pickle.dump(history.history, file)
model.save('pureTLmodel.h5')



predictions = model.predict(test_generator)
predicted_classes = predictions.argmax(axis=1)

true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())  # List of class labels
loss, accuracy = model.evaluate(test_generator)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")


cm = confusion_matrix(true_classes, predicted_classes)
print(classification_report(true_classes, predicted_classes, target_names=class_labels))


