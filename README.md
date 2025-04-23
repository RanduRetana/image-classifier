# image-classifier
# Clasificación de Imágenes de Ronaldo, Maria Sharapova y Kobe Bryant

## Descripción del Proyecto

Este proyecto tiene como objetivo entrenar un modelo de deep learning capaz de clasificar imágenes de tres personas: **Cristiano Ronaldo**, **Maria Sharapova** y **Kobe Bryant**. A través de redes neuronales convolucionales (CNN), buscamos que el modelo identifique a cada individuo a pesar de variaciones como ropa, fondo y pose.

Uno de los principales **retos** fue trabajar con un **dataset limitado**, tanto en cantidad como en diversidad de imágenes. A diferencia de enfoques como el de **AlexNet**, que fue entrenado con millones de imágenes (ImageNet), **nuestro objetivo es lograr alta precisión utilizando una base de datos pequeña y recursos computacionales reducidos**.

Durante el desarrollo, se identificaron problemas de **sesgo visual** en el dataset inicial, lo que motivó ajustes en la base de datos y mejoras progresivas en el modelo.

---

## Descripción del Dataset

La carpeta `data_images/` contiene tres subcarpetas:

- `Ronaldo/`
- `Maria Sharapova/`
- `Kobe Bryant/`

Inicialmente, las imágenes de Ronaldo tenían un sesgo importante: **todas mostraban al jugador con uniforme rojo**. Esto provocaba que el modelo confundiera a Sharapova o Kobe cuando aparecían con ropa de ese mismo color. Para solucionar esto:

- Se añadieron imágenes de Ronaldo **con ropa de distintos colores**.
- Se incorporaron **más imágenes de los tres sujetos** para mejorar la diversidad y cantidad de datos por clase.

---

## Evolución del modelo


### Modelo Inicial (V1.0)

Una CNN simple con tres capas convolucionales y una capa densa final:


Input: Imagen RGB de 128x128
↓ Conv2D (32) → ReLU
↓ Conv2D (64) → ReLU
↓ Conv2D (128) → ReLU
↓ Flatten
↓ Dense (128) → ReLU
↓ Dense (3) → Softmax



Este modelo sirvió como primera aproximación, pero presentaba limitaciones en capacidad de generalización y requería muchas épocas para converger. Además, no incluía técnicas de regularización como MaxPooling o Dropout.

En esta versión, tras 150 epochs, se alcanzaba una accuracy de 0.7824, con loss de 199.0584, resultados que evidencian una arquitectura sencilla y con un gran margen de mejora.

---

### Modelo Mejorado (V2.0) (Inspirado en AlexNet)

Basado en el análisis del paper ["ImageNet Classification with Deep Convolutional Neural Networks" (Krizhevsky et al., 2012)](https://papers.nips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf), se rediseñó el modelo incorporando elementos clave de **AlexNet**, como MaxPooling y Dropout, además, el tamaño de las imágenes se cambió a 224x224.

```python
model = Sequential()

model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(3, activation='softmax'))
```
Este rediseño mejoró significativamente la precisión y redujo la cantidad de épocas necesarias para converger. En 100 épocas, el nuevo modelo logró un rendimiento mucho mayor que el modelo original, alcanzando una accuracy de 0.8705 y loss de 0.45

El artículo de Krizhevsky et al. sirvió como justificación teórica para el uso de arquitecturas profundas, ReLU como función de activación, y técnicas de regularización como Dropout y MaxPooling. La arquitectura AlexNet demostró ser altamente efectiva en clasificación de imágenes a gran escala, validando nuestro enfoque.

---

### Modelo Mejorado Con Transfer Learning (V3.0)

Para la tercera versión, se utilizó MobileNetV2, una arquitectura ligera optimizada para eficiencia, basada en el paper:
[Sandler, Mark, et al. "MobileNetV2: Inverted Residuals and Linear Bottlenecks" (CVPR 2018)]

Se congelaron las capas convolucionales preentrenadas en ImageNet y se añadieron capas densas personalizadas para nuestro problema de tres clases.

Resultados con esta versión (tras 40 épocas):

Accuracy: 0.9389

Loss: 0.2555

Gracias al uso de transfer learning, se logró mejorar el rendimiento sin necesidad de grandes volúmenes de datos.

### Generación de Datos de Entrenamiento y Testeo
Se utilizaron generadores de imágenes con ImageDataGenerator de Keras para:

- Redimensionar imágenes a 224x224 px

- Normalizar valores de píxeles

- Aplicar data augmentation (giro, zoom, traslación) para aumentar la robustez del modelo

- Los datos fueron separados en 80% entrenamiento y 20% testeo.

---

### Métricas de Evaluación
- Accuracy (precisión global)

- Matriz de Confusión

- [Futuro] Se planea incorporar F1-score, Precision y Recall para análisis más profundo del rendimiento, especialmente si se detectan desbalances.

---

### Conclusiones
- El dataset inicial presentaba sesgos importantes que fueron corregidos agregando imágenes más variadas.

- El modelo inicial fue útil como prueba de concepto, pero fue superado ampliamente por la versión inspirada en AlexNet, la cual logró mejor rendimiento con menos entrenamiento.

- El artículo de Krizhevsky et al. sirvió como inspiración y respaldo para mejorar la arquitectura y demostrar la eficacia de redes profundas en clasificación visual.



