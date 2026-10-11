# CNN OCR en C/C++ para MNIST

Implementación de una red neuronal convolucional (CNN) para reconocer dígitos manuscritos del 0 al 9 a partir de imágenes de **28 x 28 píxeles**. El proyecto está escrito en C/C++ y contiene tres variantes del mismo acelerador, preparadas para estudiar distintos compromisos entre simplicidad, paralelismo y latencia en flujos de síntesis de alto nivel (HLS).

<p align="center">
  <strong>Convolución 2D</strong> &nbsp;&rarr;&nbsp; <strong>ReLU</strong> &nbsp;&rarr;&nbsp; <strong>Max-pooling 2 x 2</strong> &nbsp;&rarr;&nbsp; <strong>Capa totalmente conectada</strong> &nbsp;&rarr;&nbsp; <strong>Softmax</strong>
</p>

## Características

- Clasificación de dígitos basada en el conjunto de datos [MNIST](http://yann.lecun.com/exdb/mnist/).
- Entrada de un canal con formato `28 x 28` y valores `float` normalizados.
- Una capa convolucional `3 x 3` con 32 filtros y stride 1.
- Activación ReLU después de la convolución.
- Max-pooling `2 x 2` con stride 2.
- Capa totalmente conectada de `5408` entradas a `10` clases.
- Salida en forma de probabilidades mediante softmax.
- Testbench que recorre automáticamente las carpetas de clases y calcula exactitud, errores y tiempo total.
- Directivas HLS para comparar diferentes estrategias de optimización.

## Arquitectura de la red

| Etapa | Forma de entrada | Configuración | Forma de salida |
| --- | ---: | --- | ---: |
| Entrada | `28 x 28 x 1` | Imagen MNIST | `28 x 28 x 1` |
| Convolución + ReLU | `28 x 28 x 1` | Kernel `3 x 3`, 32 filtros, stride 1 | `26 x 26 x 32` |
| Max-pooling | `26 x 26 x 32` | Ventana `2 x 2`, stride 2 | `13 x 13 x 32` |
| Flatten | `13 x 13 x 32` | Reordenamiento lineal | `5408` |
| Fully connected | `5408` | 10 salidas | `10` |
| Softmax | `10` | Probabilidades por clase | `10` |

La clase predicha es el índice de la mayor probabilidad de salida.

## Variantes de implementación

Las tres implementaciones comparten la misma interfaz `top_cnn(...)` y la misma estructura matemática. Se diferencian en las directivas de síntesis:

| Archivo | Enfoque |
| --- | --- |
| `src/Arquitectura Base.c` | Referencia funcional, sin optimizaciones HLS específicas. |
| `src/Arquitectura Iterativa.c` | Ejecución explícitamente conservadora mediante directivas que desactivan pipeline y unrolling. |
| `src/Arquitectura Baja Latencia.c` | Particionado de arrays, pipeline y unrolling para favorecer el paralelismo y reducir la latencia. |

Estas variantes no deben compilarse juntas: todas definen las mismas funciones (`conv`, `pool`, `fc` y `top_cnn`).

## Estructura del repositorio

```text
.
├── lib/
│   ├── activations.h          # ReLU y softmax
│   ├── params.h               # Dimensiones y constantes de la CNN
│   └── top_cnn.h              # Interfaz del acelerador
├── src/
│   ├── Arquitectura Base.c
│   ├── Arquitectura Iterativa.c
│   └── Arquitectura Baja Latencia.c
├── testbench/
│   ├── main.cpp               # Evaluación sobre el dataset
│   ├── conv_weights.h         # Pesos y bias de convolución
│   ├── fc_weights.h           # Pesos y bias de la capa fully connected
│   ├── input_image.txt        # Imagen individual de ejemplo
│   └── mnist_txt_1000/        # Imágenes organizadas en carpetas 0..9
└── tools/
    └── train_fc.py            # Regeneración reproducible de los pesos FC
```

## Requisitos

### Simulación software

- Windows, Linux o macOS.
- Compilador compatible con C++17, por ejemplo [GCC](https://gcc.gnu.org/) o [Clang](https://clang.llvm.org/).
- Python 3.10 o superior con `numpy` y `scikit-learn` para regenerar los pesos.
- Los pesos de convolución y de la capa totalmente conectada incluidos en `testbench/`.

### Síntesis HLS

Para explorar la implementación en hardware se recomienda [AMD Vitis HLS](https://www.amd.com/en/products/software/adaptive-socs-and-fpgas/vitis.html) o una herramienta Xilinx/Vitis compatible con las directivas `#pragma HLS` incluidas en el código.

## Compilación y ejecución

Desde la raíz del proyecto, la variante base puede compilarse con:

```powershell
New-Item -ItemType Directory -Force build | Out-Null
g++ -std=c++17 -Ilib -Itestbench `
  testbench/main.cpp `
  "src/Arquitectura Base.c" `
  -o build/cnn_base.exe
```

El testbench acepta la ruta del dataset como primer argumento. Si no se proporciona, usa `mnist_txt_1000` relativo al directorio actual:

```powershell
Set-Location testbench
..\build\cnn_base.exe mnist_txt_1000
```

Para compilar otra variante, sustituya el archivo de arquitectura:

```powershell
g++ -std=c++17 -Ilib -Itestbench `
  testbench/main.cpp `
  "src/Arquitectura Baja Latencia.c" `
  -o build/cnn_low_latency.exe
```

El programa muestra:

- número total de imágenes evaluadas;
- predicciones correctas;
- exactitud global;
- tiempo total de ejecución;
- ruta, predicción y etiqueta real de cada imagen mal clasificada.

### Regenerar los pesos

Los pesos de `testbench/fc_weights.h` ya están incluidos para permitir una ejecución inmediata. Si se modifican los datos o se desea regenerarlos, instala las dependencias y ejecuta:

```powershell
python -m pip install numpy scikit-learn
python tools/train_fc.py
```

El script reutiliza los pesos convolucionales del proyecto, reproduce la convolución y el max-pooling, entrena la capa final y exporta el header consumido por el testbench.

## Formato de los datos

El testbench espera esta organización:

```text
mnist_txt_1000/
├── 0/
│   ├── 0.txt
│   └── ...
├── 1/
├── ...
└── 9/
```

Cada archivo `.txt` debe contener **784 valores `float`**, separados por comas, correspondientes a una imagen de `28 x 28`. Las carpetas representan la etiqueta correcta.

## Estado actual y próximos pasos

El código fuente, el dataset de evaluación y los pesos entrenados están incluidos. La ejecución validada sobre las 1000 imágenes disponibles alcanza actualmente **100% de exactitud** en este banco. El PDF del proyecto documenta una ejecución previa de **98,8%**; ambas cifras dependen del conjunto de datos y del proceso de entrenamiento utilizado. Tampoco se incluye todavía un proyecto de Vitis HLS ni un sistema de build multiplataforma.

Una evolución natural del proyecto sería:

1. Incorporar los pesos completos y documentar su procedencia y entrenamiento.
2. Añadir CMake o un workflow de GitHub Actions para automatizar la compilación.
3. Añadir validación más estricta del formato y tamaño de las imágenes de entrada.
4. Añadir pruebas unitarias para cada etapa de la CNN y métricas reproducibles de exactitud, latencia y recursos HLS.

## Tecnologías y referencias

- [C++](https://isocpp.org/) y [GCC](https://gcc.gnu.org/)
- [Vitis HLS](https://docs.amd.com/r/en-US/ug1399-vitis-hls)
- [MNIST Database](http://yann.lecun.com/exdb/mnist/)
- [Informe técnico del proyecto](Reconocimiento%20de%20dígitos%20con%20redes%20neuronales%20en%20Vitis%20HLS.pdf)
- [Open Neural Network Exchange (ONNX)](https://onnx.ai/) como referencia para futuros flujos de exportación de modelos

## Licencia

No se ha definido una licencia para este repositorio. Antes de publicarlo, añade un archivo `LICENSE` con la licencia que corresponda al proyecto y a los pesos del modelo.