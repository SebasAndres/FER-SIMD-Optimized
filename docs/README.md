# Optimized Facial Expression Recognition with SIMD 
#### Proyecto final Arquitectura y Organización del Computador - FCEN UBA :wrench:

### Abstract 
Este proyecto es un clasificador de caras construido en base a `OpenCV`, `HaarExtractionAlgorithm` y 
`KNN`.

<img src='img/flujo.png' width=500 height=500>

Mi idea es implementar un sistema de Face Emotion Recognition (FER) optimizado con SIMD que detecte en tiempo real caras y las clasifique según su estado de ánimo (feliz, triste, enojado, neutral, etcétera). El entregable sería un proyecto el cual vos al correrlo te prende la cámara de la computadora y ves cómo clasifica todas las caras que aparezcan.

Como contexto, el trabajo en FER tiene 3 etapas principales:

> [1] Detección de caras en las imágenes.
> [2] Extracción de features de las caras.
> [3] Algoritmo de clasificación.

[1] **Detección de caras en las imágenes**: Para la primera parte, la idea es implementarlo con Haar Features. Aunque no profundicé demasiado esta implementación todavía (estoy usando  un modelo Haar entrenado para testear) el cálculo de estos features se hace con promedios de intensidades de píxeles, algo que es completamente realizable con SIMD.

[2] **Extracción de features de las caras**: Una vez recortada la imagen de la cara, para vectorizar esos píxeles puedo usar Principal Component Analysis (PCA) o Local Binary Patterns (LBP). Por un lado, PCA es una técnica que estoy viendo en la materia ALC y sé que se usó en proyectos finales anteriores, entonces es algo realizable. LBP, es otra técnica que se menciona en los papers ya que es un poco mejor computacionalmente y no la vi implementada en otro lado. Podría probar cuál da mejores resultados.

[3] **Clasificación de caras**: El approach inicial, y vi en los papers que es muy usado, va a ser con KNN. Sería computar la matriz de distancias entre vectores y hacer la inferencia de forma optimizada con SIMD.

## TODO:
- Trabajar en que funcione `application.cpp` usando de forma modularizada a `FaceExtractor` y `FaceClassifier`. 
--> Usar **KNN** y **PCA** en primer lugar.
- Luego trabajar en `FaceExtractor::LBPExtractor`.

#### Run application
<img src='img/application_loop.png' width=200 height=300>

### Papers
* [Haar Features | Viola & Jones Algorithm](https://www.merl.com/publications/docs/TR2004-043.pdf)
* [Facial expression recognition techniques](https://ietresearch.onlinelibrary.wiley.com/doi/epdf/10.1049/iet-ipr.2018.6647)
* [LBP](https://link.springer.com/chapter/10.1007/978-3-540-24670-1_36)

