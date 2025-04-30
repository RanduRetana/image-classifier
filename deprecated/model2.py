import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

base_dir = "IA/data_images"

# Para ResNet50, debemos usar preprocess_input en lugar de rescale
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,  # Usar el preprocesamiento específico de ResNet50
    rotation_range=20,  
    width_shift_range=0.2,
    height_shift_range=0.2,  
    zoom_range=0.3,
    shear_range=0.1,  
    horizontal_flip=True,
    validation_split=0.2
)

# Generador para datos de entrenamiento
train_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=(224, 224),
    batch_size=16,  
    class_mode='categorical',
    subset='training',
    shuffle=True  
)

# Generador para datos de validación
val_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical',
    subset='validation'
)

print("Class indices:", train_generator.class_indices)


# Cargar ResNet50 preentrenada
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Congelar inicialmente

# Construir el modelo completo
x = base_model.output
x = GlobalAveragePooling2D()(x)
# Primera capa densa más grande
x = Dense(512, activation='relu')(x)
x = Dropout(0.4)(x)
# Segunda capa densa
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
# Capa de salida con 3 clases
predictions = Dense(3, activation='softmax')(x)

model_resnet = Model(inputs=base_model.input, outputs=predictions)

# Compilar el modelo
model_resnet.compile(
    optimizer=Adam(learning_rate=0.00005),  
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks para el entrenamiento
callbacks = [
    EarlyStopping(
        monitor='val_accuracy',
        patience=7,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
]

# Entrenamiento inicial
history = model_resnet.fit(
    train_generator,
    epochs=15,
    validation_data=val_generator,
    callbacks=callbacks,
    verbose=1
)

# fine-tuning

# Descongelar parte del modelo base
base_model.trainable = True

# Congelar las primeras capas y dejar las últimas para entrenar
for layer in base_model.layers[:100]:
    layer.trainable = False

# Recompilar con tasa de aprendizaje más baja para fine-tuning
model_resnet.compile(
    optimizer=Adam(learning_rate=1e-6),  # Tasa de aprendizaje mucho más baja para fine-tuning
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Segundo entrenamiento (fine-tuning)
history_finetune = model_resnet.fit(
    train_generator,
    epochs=20,
    validation_data=val_generator,
    callbacks=callbacks,
    verbose=1
)

# Evaluar el modelo
evaluation = model_resnet.evaluate(val_generator)
print(f"\nEvaluación final - Loss: {evaluation[0]:.4f}, Accuracy: {evaluation[1]:.4f}")

# Combinar historiales para visualización completa
combined_acc = history.history['accuracy'] + history_finetune.history['accuracy']
combined_val_acc = history.history['val_accuracy'] + history_finetune.history['val_accuracy']
combined_loss = history.history['loss'] + history_finetune.history['loss']
combined_val_loss = history.history['val_loss'] + history_finetune.history['val_loss']

# Graficar accuracy
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(combined_acc, label='Entrenamiento')
plt.plot(combined_val_acc, label='Validación')
plt.title('Accuracy de ResNet50')
plt.xlabel('Época')
plt.ylabel('Accuracy')
plt.legend()

# Graficar loss
plt.subplot(1, 2, 2)
plt.plot(combined_loss, label='Entrenamiento')
plt.plot(combined_val_loss, label='Validación')
plt.title('Loss de ResNet50')
plt.xlabel('Época')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Guardar el modelo
model_resnet.save('resnet50_classifier.h5')
print("Modelo guardado como 'resnet50_classifier.h5'")