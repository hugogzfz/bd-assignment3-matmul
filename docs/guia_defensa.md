# Guía de defensa – Assignment 3 (multiplicación paralela con OpenMP)

Guía corta para preparar la defensa oral. Todos los números salen de `results/` (ver `results/summary_*.csv`).

## 1. Qué hace cada fichero

| Fichero | Qué hace |
|---|---|
| `src/c/matmul3.c` | El programa medido. Tres métodos: `seq` (kernel blocked de A2, b = 96), `omp` (el mismo kernel, cada hilo calcula un bloque contiguo de filas de C) e `ijk` (el triple bucle de A1, solo como referencia). Cronometra solo la llamada al kernel, informa del número real de hilos y de `ru_maxrss`. |
| `scripts/gen_inputs.py` | Regenera las matrices de A1/A2 (semilla `[20261012, n, w]`) y comprueba su SHA-256 con `data/manifest.json`. |
| `scripts/test_correctness.py` | 351 comprobaciones: casos con respuesta conocida, identidad, ceros, comparación con NumPy (tolerancia 2γₙ), igualdad **bit a bit** entre `seq` y `omp` para todos los p, número de hilos real. Se ejecuta sobre 4 binarios (1404 en total). |
| `scripts/run_benchmarks.py` | Driver: un proceso por configuración, 1 calentamiento + 5 medidas, orden aleatorio con semilla, pausa de 3 s, presupuestos (120 s / 4 GiB), registro de `CPU_Speed_Limit`, valida cada producto y compara su SHA-256 con el de `seq`. |
| `scripts/env_info.py` | Guarda CPU, núcleos, cachés, RAM, SO, compilador, libomp… en `results/environment.json`. |
| `scripts/vec_evidence.py` | Evidencia de vectorización: avisos de clang (`-Rpass=loop-vectorize`) y recuento de instrucciones en el desensamblado (`otool -tv`). |
| `scripts/analyze.py` | Lee los CSV crudos y genera resúmenes, tablas LaTeX y figuras. |
| `Makefile` | `build`, `build-vec`, `inputs`, `test`, `bench-quick`, `bench`, `vecinfo`, `analyze`, `report`. |
| `results/raw_measurements.csv` | Todas las muestras (calentamientos incluidos). `results/runs.csv`: una fila por proceso. |
| `report/` | Fuentes del informe; `figures/` las figuras. |

## 2. Decisiones de diseño (y por qué)

1. **Baseline equivalente = kernel blocked de A2**, no el ijk. El enunciado exige comparar con el *mismo* algoritmo; el ijk solo aparece como referencia.
2. **Reparto por filas contiguas, no por bloques de 96 filas.** Con n = 1000 solo hay 11 franjas de bloques: con 8 hilos algunos harían 2 y otros 1. Por filas, cada hilo recibe n/p ± 1 filas.
3. **Sin condiciones de carrera:** cada elemento de C lo escribe un único hilo; A y B solo se leen. No hace falta `critical` ni `atomic`; la única sincronización es la barrera implícita al final del `parallel`.
4. **Resultados bit a bit idénticos:** cada C[i][j] suma los k en el mismo orden con cualquier p, y `-ffp-contract=off` impide FMA. Por eso puedo exigir igualdad exacta, no solo tolerancia.
5. **Mismo binario para `seq` y `omp`**, mismas flags (`-O2 -ffp-contract=off`), mismas entradas (SHA-256 iguales a A2).
6. **Control de hilos:** `OMP_NUM_THREADS=p`, `OMP_DYNAMIC=false`; el programa comprueba tamaño de equipo y hilos distintos del SO (`pthread_threadid_np`). Ninguna librería numérica dentro del kernel; NumPy (solo validación) con `OPENBLAS_NUM_THREADS=1`.
7. **Afinidad:** macOS no permite fijar hilos a núcleos → limitación documentada.
8. **Frontera de tiempo:** dentro, la puesta a cero de C y el fork/join; fuera, lectura de ficheros, reserva de C, checksum y validación.
9. **Protocolo:** calentamiento excluido, 5 repeticiones (3 en n = 4000), mediana e IQR, S y E con medianas, lotes para n pequeño, orden aleatorio, pausa, segunda pasada de n = 2000 para ver la deriva.
10. **Memoria:** `ru_maxrss` (en macOS en **bytes**), pico de memoria residente del proceso.

## 3. Resultados clave (memorízalos)

- 110 procesos, todos `ok`, todos correctos y **bit a bit idénticos** a `seq`. Campaña de 21 min sin throttling (`CPU_Speed_Limit` = 100).
- `seq`: ≈ 6 GFLOP/s en todos los tamaños; n = 4000 tarda 19,6 s y usa 367 MiB.
- **S(4) = 3,40–3,95, E(4) = 0,85–0,99** para todos los n. Con 8 hilos (Hyper-Threading) S cambia entre −15 % y +6 %; con 16 hilos no hay ganancia sistemática.
- **Incluso n = 100 (0,31 ms) gana: S(4) = 3,54.** H4 rechazada.
- `omp` con p = 1 ≈ `seq` (−3 % a +2 % en 6 de 8 tamaños; en los otros 2, `omp` fue *más rápido*, por ruido del baseline).
- Superlinealidad aparente (E > 1 en n = 1500) → explicada: el `seq` de ese n fue lento; respecto a `omp` p=1, E ≤ 1.
- Memoria = 3·8n² + < 1,5 MiB, no depende de p.
- Segunda pasada n = 2000: entre 3 % y 17 % más rápida que la primera (variabilidad entre periodos), pero S cambió poco (S(4) 3,50 → 3,58).
- Dentro de un proceso largo con p ≥ 3 el tiempo sube 4–8 % (n = 3000/4000); con 1 hilo no → compatible con que baje la frecuencia turbo.
- Vectorización: SSE2 2,2–2,9× y AVX2 3,9–5,1× más rápido que escalar al mismo p; AVX2 + 4 hilos = 43,9 GFLOP/s (14,9× el escalar secuencial). Hyper-Threading ayuda al código escalar (+23 %) pero no al AVX2 (−5 %).

## 4. Preguntas probables y respuestas

1. **¿Por qué el speedup se queda en ~3,5–4 y no llega a 8?**
   Hay 4 núcleos físicos. Los 2 hilos de Hyper-Threading de un núcleo comparten sus unidades de coma flotante, y el kernel ya las ocupa casi del todo. Además, con 4 núcleos activos el turbo es más bajo que con uno.

2. **¿Cómo sabes que no hay condiciones de carrera?**
   Por diseño, cada hilo escribe solo sus filas de C y A y B son de solo lectura. Lo compruebo con igualdad bit a bit frente a `seq` para p = 1, 2, 3, 4, 7, 8 y 16 en los tests, y en las 110 ejecuciones de la campaña.

3. **¿Por qué los resultados son idénticos bit a bit y no solo "parecidos"?**
   Porque el orden de suma de cada elemento es el mismo con cualquier p, y no hay FMA (`-ffp-contract=off`). Paralelizar por filas no reordena las sumas; paralelizar el bucle k sí lo haría.

4. **¿Cómo controlas que el número de hilos es el real?**
   Con `OMP_NUM_THREADS` y `OMP_DYNAMIC=false`. El programa cuenta el tamaño del equipo, los hilos que recibieron filas y los ids de hilo del SO distintos. Coincidieron con p en las 88 ejecuciones `omp`.

5. **¿Qué es la sobresuscripción y qué pasó?**
   Usar más hilos que contextos hardware (16 > 8). No dio ganancia consistente. A n = 100 perdió un 26 %, porque cada hilo tiene ~20 µs de trabajo y el reparto de tiempo y la barrera pesan.

6. **¿Por qué incluyes `omp` con 1 hilo?**
   Para medir el overhead del runtime de OpenMP. Salió dentro del ruido (±5 % en 6 de 8 tamaños, sin overhead positivo > 2 %).

7. **¿Cómo explicas E > 1 en n = 1500?**
   No es superlinealidad real. El `seq` de ese n fue un 7 % más lento que `omp` con p = 1, que ejecuta el mismo código. Calculando S frente a `omp` p=1, E ≤ 1. Es variabilidad entre procesos.

8. **¿Por qué mediana e IQR?**
   La mediana es robusta a muestras atípicas (interrupciones del SO) y el IQR mide la dispersión sin depender de los extremos. S y E se calculan con medianas, como pide el enunciado.

9. **¿Qué incluye el cronómetro?**
   Solo la llamada a la multiplicación: la puesta a cero de C y el fork/join de OpenMP. La lectura de ficheros, la reserva de C, el checksum y la validación quedan fuera.

10. **¿Por qué la segunda pasada salió más rápida?**
    No lo sé con certeza. No hubo límite térmico y la parte lenta fue el principio de la campaña, no el final, así que no es calentamiento progresivo. Lo documento como limitación: los tiempos absolutos tienen unos ±10 % de incertidumbre, pero los speedups apenas cambiaron.

11. **¿Cómo demuestras que hay vectorización?**
    clang dice "vectorized loop (width 2)" en SSE2 y "(width 4)" en AVX2 para la línea 101, y "not vectorized" en la versión escalar. El desensamblado muestra `mulpd/addpd` (xmm) y `vmulpd/vaddpd` (ymm) frente a `mulsd/addsd`. No hay FMA en ninguna versión.

12. **¿Cómo separas el efecto de SIMD del de los hilos?**
    Comparo builds distintos con el mismo p (ganancia SIMD) y el mismo build con distinto p (speedup de hilos). Es una tabla de build × p.

13. **¿Por qué AVX2 gana más de 4× si procesa 4 doubles?**
    Porque además desenrolla el bucle (interleave 4), lo que reparte el coste del control del bucle y de las direcciones entre más operaciones. Es una explicación plausible; no la medí con contadores.

14. **¿Por qué usaste b = 96 si b = 64 fue más rápido aquí?**
    b = 96 se fijó antes de medir, para ser comparable con A2. El bsweep es una comprobación: cambia el tiempo un 5–11 %, pero S(4) se mantiene entre 3,54 y 3,86.

15. **¿Cuándo compensa paralelizar?**
    En este kernel y con esta frontera de tiempo, desde el tamaño más pequeño medido (n = 100). Lo óptimo es usar tantos hilos como núcleos físicos (4). El umbral real queda por debajo de 100 o en llamadas aisladas en frío, que no medí.

## 5. Limitaciones que debes mencionar tú

Un solo portátil, hilos sin fijar a núcleos, sin contadores hardware ni medida de frecuencia (`powermetrics` necesita sudo), ±10 % de variabilidad entre periodos, solo 5 muestras por configuración, y el overhead de la primera llamada (creación de hilos) no analizado.

## 6. Uso de IA

Usé Claude Code (Anthropic) para el plan, el código, los scripts, el borrador del informe y esta guía, como declaro en el informe. Revisé el código y supervisé la campaña. Todos los números salen de los CSV. Las referencias están comprobadas; la página final (15) de Marr et al. está deducida del índice de la revista.
