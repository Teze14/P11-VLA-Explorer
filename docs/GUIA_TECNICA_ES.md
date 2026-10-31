# Guía técnica en español: VLA Explorer (P11)

Esta guía explica el prototipo que está en este repositorio. Empieza por usarlo: abre la escena de la taza roja, escribe `a red cup`, ejecuta y abre los cinco pasos de **Pipeline**. Después cambia solamente el texto a `a blue cube`. Así puedes relacionar cada cálculo con algo visible.

## 1. Qué problema enseña

Un sistema *vision-language-action* (VLA) relaciona una observación visual y una instrucción con una acción. Un VLA robótico completo aprende de demostraciones con imágenes, instrucciones, estado del robot y acciones; predice movimientos con una definición física. Nuestro prototipo enseña la interfaz entre esas partes: usa un modelo real de visión y lenguaje, **CLIP**, y propone una acción bidimensional con una regla matemática escrita por el equipo. No entrenamos un robot ni ejecutamos una política robótica.

CLIP ya fue entrenado antes de este proyecto para acercar representaciones de imágenes a representaciones de textos relacionados. Aquí sus pesos están **congelados**: se ejecutan para obtener resultados, pero no se actualizan con backpropagation. El modelo visual es **ViT-B/32**, un Vision Transformer. Su procesamiento interno incluye atención; la cuadrícula naranja que ves en la página es otro procedimiento, creado por nuestro código.

## 2. Recorrido exacto de una ejecución

### Paso A: entrada y cuadrícula

`toolkit/app.js` recibe una imagen y una frase. La imagen puede ser una escena dibujada con Canvas, la fotografía incluida, una imagen del usuario o una captura de cámara. Con la cuadrícula predeterminada de 3 × 3, `encodeImageRegions()` en `toolkit/model-engine.js` corta la imagen en **nueve regiones independientes**. Cada recorte se estira a 224 × 224 píxeles y el procesador de CLIP lo convierte en un tensor de forma `[1, 3, 224, 224]`: una imagen, tres canales de color y 224 × 224 píxeles.

La cuadrícula también puede ser 2 × 2 o 4 × 4. Más regiones dan detalle espacial, pero pueden partir un objeto o quitarle contexto. Los recortes **no son los parches internos** que usa ViT para procesar cada imagen.

### Paso B: los dos codificadores aprendidos

`loadModel()` carga el tokenizador, el procesador de imágenes, el codificador de texto y el codificador visual de `Xenova/clip-vit-base-patch32`. Se usa una versión ONNX cuantizada a 8 bits (`q8`) con Transformers.js y ONNX Runtime Web. Se ejecuta en el navegador mediante WebAssembly.

`encodeImageRegions()` pasa cada recorte por el codificador visual. Obtiene un vector de **512 números** por región: con 3 × 3, la tabla de vectores tiene forma `[9, 512]`. `encodeText()` divide la frase en tokens y pasa esos tokens por el codificador de texto; obtiene otro vector de 512 números, de forma `[1, 512]`. Los dos vectores se normalizan a longitud 1. Un número individual del vector no equivale por sí solo a «rojo» o «taza».

Dentro de estos codificadores Transformer hay capas que relacionan tokens, con atención y redes FFN. **Nuestro código no calcula ni muestra sus matrices de atención Q, K y V**. La visualización del proyecto empieza con los vectores finales que devuelve CLIP.

### Paso C: comparar imagen y frase

`scoreRegions()` calcula, para cada región `i`, la similitud coseno con el texto:

`s_i = (v_i · t) / (||v_i|| ||t||)`

Como los vectores ya se normalizaron, el código suma los productos de sus 512 componentes. Una puntuación mayor significa mejor correspondencia **relativa para ese modelo y ese texto**. No confirma que el objeto exista. En una ejecución guardada con `a red cup`, la región 4 obtuvo un puntaje aproximado de `0.3083`; fue el mayor de las nueve regiones.

### Paso D: softmax entre regiones

El mismo `scoreRegions()` convierte las nueve puntuaciones en nueve pesos:

`w_i = exp((s_i - max(s)) / τ) / Σ_j exp((s_j - max(s)) / τ)`

`τ` es la temperatura que controla el usuario. Se resta el máximo por estabilidad numérica; eso no cambia el resultado. Los pesos son positivos y suman 1. Con temperatura baja se concentran más en la región mejor puntuada; con temperatura alta se distribuyen más. En el ejemplo guardado, la región 4 recibió `w_4 ≈ 0.2882`, es decir, **28.82 % del peso repartido entre las nueve regiones**. No significa «28.82 % de probabilidad de que haya una taza».

Este softmax ocurre **después de CLIP**, en nuestro código. Es diferente del softmax que usan internamente las capas de atención del Transformer.

### Paso E: destino y flecha

`actionFromCrops()` en `toolkit/app.js` calcula el centro de cada región y hace un promedio ponderado:

`destino = Σ_i w_i · centro_i`

El inicio de la flecha es `(0.50, 0.88)` en coordenadas normalizadas de imagen. La propuesta de desplazamiento es:

`Δ = ganancia · (destino - inicio)`

En la ejecución guardada de `a red cup`, el destino fue aproximadamente `(0.432, 0.496)` y con ganancia 1 el desplazamiento fue `(-0.068, -0.384)`: hacia la izquierda y hacia arriba en la imagen. `+x` significa derecha y `+y` significa abajo. **No son metros, ángulos articulares ni comandos de una pinza.** Como se promedian todos los centros, el destino puede quedar entre objetos.

`draw()` usa esos pesos para colorear las regiones y dibujar la flecha sobre un Canvas. El color es una visualización de relevancia entre cada recorte y la frase. No es una explicación causal ni una segmentación del objeto.

## 3. Qué cambia cuando tocas cada control

| Cambio | Cálculo que se repite | Qué deberías observar |
|---|---|---|
| Cambiar solo `a red cup` por `a blue cube` | Nuevo vector de texto, nuevas similitudes y pesos; se reutilizan los vectores de imagen | Puede cambiar la región con mayor peso y el destino |
| Cambiar la imagen o la cuadrícula | Nuevos recortes y nuevas inferencias visuales | Cambian los vectores de imagen y posiblemente los resultados |
| Cambiar temperatura | Solo se recalcula softmax, usando similitudes ya calculadas | Cambian los pesos y posiblemente el destino |
| Cambiar opacidad | Solo se redibuja la capa de color | No cambian puntuaciones, pesos ni destino |
| Cambiar ganancia | Solo se recalcula el desplazamiento | Cambia el largo de la flecha, no el destino |
| Ocultar la región más fuerte | Se modifica la imagen con un rectángulo gris y se vuelve a ejecutar visión | Cambian los datos visuales; el nuevo resultado no garantiza localizar mejor el objeto |

Un ejemplo útil para explicar la caché: si dejas fija la imagen y cambias la frase, CLIP no necesita volver a procesar las nueve imágenes. La medición preliminar guardada registró cerca de **3 segundos** para una imagen nueva y cerca de **28 milisegundos** para una frase nueva con imagen ya procesada, en ese entorno de prueba. No prometas esos tiempos en otras computadoras.

## 4. Para qué sirve cada archivo de código

| Archivo | Responsabilidad |
|---|---|
| `toolkit/index.html` | Estructura de la interfaz: escenas, entradas, controles, panel de cinco pasos, lecciones, experimentos y fuentes. |
| `toolkit/styles.css` | Colores, fuentes, tamaños y adaptación a pantallas pequeñas. No calcula inferencia. |
| `toolkit/app.js` | Conecta botones y controles con el motor. Dibuja la escena original, la cuadrícula, el mapa de calor y la flecha; permite subir foto, capturar cámara, guardar comparaciones, ocultar una región y exportar JSON. `actionFromCrops()` está aquí. |
| `toolkit/model-engine.js` | Carga CLIP; procesa recortes y texto; normaliza vectores; calcula similitud coseno y softmax; informa formas de tensores, tiempos y uso de caché. La función `analyze()` reúne el recorrido. |
| `toolkit/vendor/` | Archivos de Transformers.js y ONNX Runtime Web que permiten ejecutar el modelo en el navegador, más sus avisos de licencia. No escribimos esas bibliotecas. |
| `toolkit/assets/` | Fotografía de ejemplo, fuentes y créditos/licencias. La escena de taza/cubo/banana se dibuja en `app.js`. |
| `toolkit/models/` | Pesos descargados para el modelo. Se generan con `download_models.py` y se excluyen de Git porque son grandes. |
| `serve.py` | Pequeño servidor HTTP local hecho con la biblioteca estándar de Python. Sirve `toolkit/` en `127.0.0.1:8767`. Debe seguir abierto mientras se usa la página. |
| `launch_windows.cmd` | En Windows, cambia a la carpeta correcta y ejecuta `python serve.py` con doble clic. |
| `download_models.py` | Descarga de Hugging Face una revisión fija del tokenizador, configuración y dos modelos ONNX; guarda tamaños y SHA-256 en un manifiesto. |
| `build_site.py` | Copia la aplicación a `dist/` para un posible alojamiento estático, sin copiar los pesos grandes. No se necesita para el uso local. |
| `requirements.txt` | Aclara que el servidor local no necesita instalar paquetes de Python. Las dependencias opcionales del notebook son aparte. |
| `notebooks/SmolVLA_inference.ipynb` | Ejercicio futuro para conectar un VLA de robótica a observaciones compatibles. **No fue ejecutado ni validado** como parte de este prototipo. |

`docs/SCIENTIFIC_SCOPE.md` fija exactamente qué afirmamos y qué no. `docs/TEACHING_GUIDE.md` propone una clase de 15 minutos. `docs/EVALUATION_PLAN.md` planea cómo medir aprendizaje, sin fingir que el estudio ya ocurrió. `docs/ROADMAP.md` distribuye trabajo futuro; `docs/SAMPLE_PAPER_NOTES.md` relaciona las lecturas de ejemplo con decisiones de diseño. `evaluation/example-run.json` contiene una ejecución guardada; `evaluation/latency-measurements.json`, mediciones de tiempo; `evaluation/functional-checks.json`, comprobaciones realizadas. `README.md` es la entrada para instalar y entender el repositorio.

## 5. Cómo lo descargan y ejecutan Robert y Michelle

**Con Git:** instalen Git, Python 3.10 o superior y un navegador Chrome o Edge reciente. En una terminal ejecuten:

```powershell
git clone https://github.com/Teze14/P11-VLA-Explorer.git
cd P11-VLA-Explorer
python download_models.py
python serve.py
```

Luego abren `http://127.0.0.1:8767/` en **esa misma computadora**. `127.0.0.1` siempre indica la máquina de quien abre el navegador: el enlace local de Jesús no abre la copia de otra persona. Para detener el servidor, `Ctrl+C` en la terminal.

**Sin Git:** entren a `https://github.com/Teze14/P11-VLA-Explorer`, pulsen **Code → Download ZIP**, descompriman el ZIP, abran una terminal dentro de la carpeta `P11-VLA-Explorer-main` y ejecuten `python download_models.py` y `python serve.py`. En Windows también pueden ejecutar `launch_windows.cmd` después de descargar los modelos.

La primera descarga requiere internet y aproximadamente **154 MB de pesos del modelo**. Es mejor hacerla antes de una presentación. La biblioteca de inferencia del navegador ya viene en el repositorio. Las imágenes y frases se procesan localmente; la descarga inicial solo obtiene los archivos del modelo. Si el puerto 8767 está ocupado, ejecuten `python serve.py --port 8768` y abran `http://127.0.0.1:8768/`. No abran `index.html` con doble clic: el modelo necesita que la página se sirva por HTTP.

## 6. Preguntas que deberías poder responder

1. **¿Qué está aprendido?** Los pesos congelados de los codificadores visual y textual de CLIP.
2. **¿Qué programamos nosotros?** La cuadrícula, el softmax entre regiones, la temperatura educativa, la regla del destino/flecha y la visualización.
3. **¿Qué significa el porcentaje de un recuadro?** Su peso relativo entre los recuadros actuales. No es una probabilidad de presencia o de éxito robótico.
4. **¿Por qué a veces la flecha no termina sobre el objeto?** Usa el promedio de todos los centros, no el centro exclusivo del recuadro ganador.
5. **¿Qué falta para un VLA robótico completo?** Observaciones compatibles, estado del robot, acciones con unidades físicas, una política entrenada con demostraciones y una evaluación en simulador o robot.

Para una demostración honesta: mantén fija la escena, compara `a red cup` con `a blue cube`, muestra los valores en **Similarity** y explica que el modelo produce la comparación imagen-texto mientras la regla del programa produce la flecha.
