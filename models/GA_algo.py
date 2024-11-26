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

# Spatial Attention Layer
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
data_path = "../crack_images"
train_datagen = ImageDataGenerator(validation_split=0.3, preprocessing_function=preprocess_input)
train_generator = train_datagen.flow_from_directory(data_path, target_size=(224, 224), batch_size=64, shuffle=True, class_mode='categorical', subset='training')
validation_datagen = ImageDataGenerator(validation_split=0.3, preprocessing_function=preprocess_input)
validation_generator = validation_datagen.flow_from_directory(data_path, target_size=(224, 224), batch_size=64, class_mode='categorical', subset='validation')

# Base model ResNet50
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False
x = base_model.output
x = SpatialAttention()(x)
# Output from the attention layer
x = layers.GlobalAveragePooling2D()(x)

# Below function creates and trains model based on genetic algorithm parameters
def create_model(attention_output, num_layers, neurons_per_layer):
    x = attention_output
    for _ in range(num_layers):
        x = layers.Dense(neurons_per_layer[_], activation='relu')(x)
        # Regularization
        x = layers.Dropout(0.5)(x)
    output_layer = layers.Dense(2, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=output_layer)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# GA Parameters
population_size = 10
generations = 5
max_layers = 5
neuron_range = (16, 1024)

# Initialize population
def initialize_population():
    return [
        {
            "num_layers": np.random.randint(1, max_layers + 1),
            "neurons": [np.random.randint(neuron_range[0], neuron_range[1]) for _ in range(max_layers)]
        }
        for _ in range(population_size)
    ]

# Fitness evaluation function
def evaluate_model(individual, attention_layer_output, train_data, val_data):
    num_layers = individual["num_layers"]
    neurons_per_layer = individual["neurons"][:num_layers]  # Select the `num_layers`
    model = create_model(attention_layer_output, num_layers, neurons_per_layer)
    history = model.fit(
        train_data,
        epochs= 5 ,  
        validation_data=val_data,
        verbose=1
    )
    # Returen max val accuracy as fitness score
    return max(history.history["val_accuracy"])

# Selection function
def select_parents(population, fitness_scores):
    sorted_population = [x for _, x in sorted(zip(fitness_scores, population), key=lambda pair: pair[0], reverse=True)]
    return sorted_population[:2]

# Crossover function
def crossover(parent1, parent2):
    child = {
        "num_layers": random.choice([parent1["num_layers"], parent2["num_layers"]]),
        "neurons": [
            random.choice([p1, p2]) for p1, p2 in zip(parent1["neurons"], parent2["neurons"])
        ]
    }
    return child

# Mutation function
def mutate(individual):
    if np.random.rand() < 0.2:  # 20% chance of mutation
        individual["num_layers"] = np.random.randint(1, max_layers + 1)
    if np.random.rand() < 0.5:  # 50% chance of mutation in neurons
        individual["neurons"] = [np.random.randint(neuron_range[0], neuron_range[1]) for _ in range(max_layers)]
    return individual

# Main GA
def genetic_algorithm(attention_layer_output, train_data, val_data, output_file='ga_output.txt'):
    # Initialize population and other parameters
    population = initialize_population()
    best_individual = None
    best_fitness = -np.inf

    # Output file
    with open("res.txt", 'w') as file:
        file.write("Genetic Algorithm Output Log\n\n")
        
        for generation in range(generations):
            file.write(f"Generation {generation + 1}/{generations}\n")
            print(f"Generation {generation + 1}/{generations}")

            # Evaluate fitness scores for the population
            fitness_scores = [
                evaluate_model(individual, attention_layer_output, train_data, val_data)
                for individual in population
            ]

            # Log fitness scores
            file.write(f"Fitness Scores: {fitness_scores}\n")

            # Track the best individual
            max_fitness_index = np.argmax(fitness_scores)
            if fitness_scores[max_fitness_index] > best_fitness:
                best_fitness = fitness_scores[max_fitness_index]
                best_individual = population[max_fitness_index]

            file.write(f"Best Fitness in Generation {generation + 1}: {best_fitness}\n")
            file.write(f"Best Individual: {best_individual}\n")

            # Select parents and produce next generation
            parents = select_parents(population, fitness_scores)
            new_population = [mutate(crossover(parents[0], parents[1])) for _ in range(population_size)]
            population = new_population
            file.write("\n")
        
        # Final optimal configuration
        file.write("\nOptimal Configuration Found:\n")
        file.write(f"Number of layers: {best_individual['num_layers']}\n")
        file.write(f"Neurons per layer: {best_individual['neurons'][:best_individual['num_layers']]}\n")
        file.write(f"Best Fitness Score: {best_fitness}\n")
    
    
genetic_algorithm(x, train_generator, validation_generator)

