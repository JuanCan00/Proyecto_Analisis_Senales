# Explicación Detallada del Algoritmo *Imitation* (`imitation.py`)

Este documento explica paso a paso el funcionamiento **matemático** y **computacional** del código `imitation.py`, describiendo cada variable, cada operación en el dominio del tiempo y la frecuencia, la construcción de la permutación biyectiva y la generación de las claves compatibles con `reverse.py` (extraído del código original de la profesora).

---

## 1. Idea General del Algoritmo

El objetivo de **Imitation** es ocultar un audio secreto (`secret`) reorganizando las frecuencias de sus **coeficientes complejos de Fourier** para que la distribución de sus picos de magnitud imite el orden de magnitudes de un audio de referencia (`target`).

Al reubicar los coeficientes complejos completos (parte real e imaginaria juntas):
- El espectro de magnitud del audio codificado imita la ubicación frecuencial de los picos de `target`.
- Las relaciones de fase originales de `secret` quedan desordenadas, por lo que en el dominio temporal el audio codificado suena como ruido ininteligible.
- **No se destruye información**: cada coeficiente complejo original de `secret` sigue existiendo intacto, solo que en otra frecuencia. Al guardar el mapa de permutación en `FFT_key_{i}.csv`, el algoritmo **Reverse** puede devolver cada coeficiente a su posición exacta y reconstruir el mensaje secreto.

---

## 2. Diccionario de Variables Principales en `imitation.py`

| Variable | Tipo / Dimensión | Significado matemático y computacional |
|---|---|---|
| `fs` | `int` (`44100`) | Frecuencia de muestreo en Hz ($f_s$). Determina la frecuencia de Nyquist $f_s / 2 = 22\,050\ \text{Hz}$. |
| `target` | `float64` de longitud $N$ | Vector temporal discreto $t[n]$ de la señal de referencia ($N = 383\,184$ muestras, $8.6890\ \text{s}$). |
| `secret` | `float64` de longitud $N$ | Vector temporal discreto $x[n]$ del mensaje secreto original ($N = 383\,184$ muestras). |
| `N` | `int` (`383184`) | Número total de muestras en el dominio temporal (número **par**). |
| `T` | `complex128` de longitud $M$ | Espectro `rfft(target)` con $M = \lfloor N/2 \rfloor + 1 = 191\,593$ coeficientes complejos. |
| `S` | `complex128` de longitud $M$ | Espectro `rfft(secret)` con $M = 191\,593$ coeficientes complejos ($S[k] = a_k + j b_k$). |
| `M` | `int` (`191593`) | Cantidad de coeficientes espectrales no redundantes devueltos por `rfft`. |
| `P` | `int64` de longitud $M - 2$ | Índices permutables $[1, 2, \dots, M - 2]$ (excluye DC $k=0$ y Nyquist $k=M-1$ cuando $N$ es par). |
| `I1` | `int64` de longitud $|P|$ | Posiciones relativas dentro de `P` que ordenan $|S[P]|$ de menor a mayor magnitud. |
| `I2` | `int64` de longitud $|P|$ | Posiciones relativas dentro de `P` que ordenan $|T[P]|$ de menor a mayor magnitud. |
| `K` | `int64` de longitud $M$ | **Clave de permutación biyectiva**: indica qué coeficiente original de `S` ocupa cada posición de `E`. |
| `E` | `complex128` de longitud $M$ | Espectro complejo reorganizado: $E[m] = S[K[m]]$. |
| `encoded` | `float64` de longitud $N$ | Señal temporal codificada obtenida mediante `irfft(E, n=N)`. |

---

## 3. Desarrollo Paso a Paso (Matemático y Computacional)

### Paso 1. Carga y conversión física exacta (`cargar_audio_mono`)
Los seis archivos originales (`target.wav` y `secret1.wav` a `secret5.wav`) son monoaurales a $f_s = 44\,100\ \text{Hz}$ con profundidad de **24 bits PCM**.

Como NumPy no tiene tipo nativo de 24 bits, `scipy.io.wavfile.read` carga cada muestra en un entero `int32` desplazado 8 bits a la izquierda (rango $[-2^{31}, 2^{31} - 256]$). La función divide entre $2^{31}$:

```python
senal = datos_crudos.astype(np.float64) / float(2 ** 31)
```

- Esto equivale exactamente a dividir muestras de 24 bits por $2^{23}$, llevando la señal al rango físico estándar $[-1.0, 1.0)$.
- **No se normaliza por el máximo** de cada audio, preservando las amplitudes relativas reales.
- Si llegara un audio estéreo (`ndim == 2`), la función promedia ambos canales en `float64` y emite una advertencia explícita.
- Luego `validar_compatibilidad` verifica que `target` y `secret` tengan exactamente la misma `fs` y el mismo `N`, evitando recortes o rellenos silenciosos.

---

### Paso 2. Transformada de Fourier para señales reales (`rfft`)
La Transformada Discreta de Fourier (DFT) transforma una señal temporal $x[n]$ de longitud $N$ al dominio de la frecuencia:

$$X[k] = \sum_{n=0}^{N-1} x[n]\, e^{-j \frac{2\pi}{N} k n}, \quad k = 0, 1, \dots, N-1$$

En `imitation.py` calculamos la DFT mediante el algoritmo rápido para señales reales **`scipy.fft.rfft`**:

```python
T = rfft(target)
S = rfft(secret)
```

**¿Por qué `rfft` y no `fft`?**
1. La **DFT** es la definición matemática; la **FFT** es el algoritmo rápido $O(N \log N)$ que la calcula exactamente.
2. Como el audio es estrictamente real ($x[n] \in \mathbb{R}$), su espectro cumple **simetría conjugada**: $X[N - k] = X^*[k]$.
3. Por esa simetría, la mitad superior del espectro es redundante. `rfft` devuelve solo los $M = \lfloor N/2 \rfloor + 1 = 191\,593$ coeficientes de frecuencias no negativas $f_k = k \cdot \frac{f_s}{N} \in [0, 22\,050\ \text{Hz}]$ (con resolución $\Delta f \approx 0.1151\ \text{Hz}$).
4. Al permutar únicamente dentro de estos $M$ coeficientes y reconstruir con `irfft`, se garantiza automáticamente una señal temporal real.

Cada coeficiente complejo $S[k] = a_k + j b_k = |S[k]| e^{j \phi[k]}$ contiene conjuntamente:
- **Magnitud** $|S[k]| = \sqrt{a_k^2 + b_k^2}$ (intensidad de esa frecuencia).
- **Fase** $\phi[k] = \operatorname{atan2}(b_k, a_k)$ (alineación temporal de la onda, esencial para la inteligibilidad de la voz).

---

### Paso 3. Selección de índices permutables `P` (DC y Nyquist fijos)
En `obtener_indices_permutables`:

```python
if num_muestras % 2 == 0:
    return np.arange(1, longitud_espectro - 1, dtype=np.int64)
return np.arange(1, longitud_espectro, dtype=np.int64)
```

- El coeficiente **DC** ($k = 0$, $f = 0\ \text{Hz}$) es $\sum_{n=0}^{N-1} x[n]$, siempre **estrictamente real** ($\operatorname{Im} = 0$).
- Cuando $N$ es **par** ($N = 383\,184$), el coeficiente de **Nyquist** ($k = M - 1 = N/2$, $f = f_s/2$) es $\sum_{n=0}^{N-1} (-1)^n x[n]$, también **estrictamente real** ($\operatorname{Im} = 0$).
- Si moviéramos un coeficiente complejo $a + jb$ ($b \neq 0$) hacia $k = 0$ o $k = M - 1$, `irfft` descartaría automáticamente su parte imaginaria $b$ para producir una señal real, **causando pérdida irreversible**. Por eso `P = [1, 2, ..., 191591]` deja fijos ambos extremos.

---

### Paso 4. Ordenamiento de magnitudes y construcción de la permutación `K`
El corazón del algoritmo sigue exactamente el pseudocódigo oficial:

```python
I1 = np.argsort(np.abs(S[P]))
I2 = np.argsort(np.abs(T[P]))

K = np.arange(M, dtype=np.int64)
K[P[I2]] = P[I1]

E = S[K]
```

1. **`np.abs(S[P])` y `np.abs(T[P])`**: calculan las magnitudes de los coeficientes permutables. Se usan **solo para ordenar**, nunca para reemplazar los valores complejos.
2. **`I1 = np.argsort(...)` e `I2 = np.argsort(...)`**: obtienen los índices que ordenan las magnitudes de menor a mayor dentro de `P`. Así:
   - `P[I1[0]]` es el *bin* donde `Secret` tiene su menor magnitud; `P[I1[-1]]` es donde tiene su mayor magnitud.
   - `P[I2[0]]` es el *bin* donde `Target` tiene su menor magnitud; `P[I2[-1]]` es donde tiene su mayor magnitud.
3. **`K = np.arange(M)`**: inicializa la clave como la identidad $[0, 1, \dots, M-1]$, garantizando que $K[0] = 0$ y $K[M - 1] = M - 1$ permanezcan fijos.
4. **`K[P[I2]] = P[I1]`**: establece la correspondencia rango a rango:
   $$K\big[P[I_2[r]]\big] = P[I_1[r]], \quad r = 0, 1, \dots, |P| - 1$$
   Es decir, en la frecuencia donde `Target` tiene su $r$-ésima magnitud ordenada, colocaremos el coeficiente que `Secret` tenía en su $r$-ésima magnitud ordenada.
5. **Demostración de biyectividad (`validar_clave_biyectiva`)**: como `I1` e `I2` son biyecciones sobre $\{0, \dots, |P|-1\}$, cada índice de `P` aparece exactamente una vez en `P[I1]` y una vez en `P[I2]`. Sumando los índices fijos DC y Nyquist, `K` contiene todos los enteros de $0$ a $M - 1$ sin repetirse ni faltar ninguno.
6. **`E = S[K]`**: reorganiza los **coeficientes complejos completos** (magnitud y fase simultáneamente): $E[m] = S[K[m]]$.

---

### Paso 5. Ejemplo numérico pequeño ($N = 8$, $M = 5$)
Para visualizar exactamente cómo operan los índices, supongamos una señal par de longitud $N = 8$ ($M = 5$ coeficientes en `rfft`: índices $0, 1, 2, 3, 4$).

- Fijos: $k = 0$ (DC) y $k = 4$ (Nyquist).
- Permutables: $P = [1, 2, 3]$.

| Índice $k$ | $S[k]$ (Secret) | $|S[k]|$ | $T[k]$ (Target) | $|T[k]|$ |
|:---:|:---:|:---:|:---:|:---:|
| 0 (DC) | $2 + 0j$ | 2 | $1 + 0j$ | 1 |
| **1** | $3 + 4j$ | **5** | $0 + 2j$ | **2** (menor) |
| **2** | $0 - 1j$ | **1** | $6 + 8j$ | **10** (mayor) |
| **3** | $-6 + 8j$ | **10** | $0 - 4j$ | **4** (media) |
| 4 (Nyq) | $-0.5 + 0j$ | 0.5 | $0.2 + 0j$ | 0.2 |

1. Orden ascendente en `P = [1, 2, 3]`:
   - En Secret (`[5, 1, 10]`): `I1 = [1, 0, 2]` $\implies$ `P[I1] = [2, 1, 3]`.
   - En Target (`[2, 10, 4]`): `I2 = [0, 2, 1]` $\implies$ `P[I2] = [1, 3, 2]`.
2. Construcción de `K` (`K[[1, 3, 2]] = [2, 1, 3]`):
   - `K[0] = 0`, `K[1] = 2`, `K[2] = 3`, `K[3] = 1`, `K[4] = 4` $\implies$ **`K = [0, 2, 3, 1, 4]`**.
3. Permutación compleja `E = S[K]`:
   - `E = [S[0], S[2], S[3], S[1], S[4]]` $= [2+0j,\; 0-1j,\; -6+8j,\; 3+4j,\; -0.5+0j]$.
   - Sus magnitudes en `P` son $[1, 10, 5]$, exactamente con el pico máximo en $k = 2$ igual que `Target`, conservando los complejos exactos de `Secret`.
4. **Recuperación exacta en `Reverse` (`X_recovered[K] = E`)**:
   - `X_recovered[K[0]] = E[0]` $\to$ `X_recovered[0] = S[0]`
   - `X_recovered[K[1]] = E[1]` $\to$ `X_recovered[2] = S[2]`
   - `X_recovered[K[2]] = E[2]` $\to$ `X_recovered[3] = S[3]`
   - `X_recovered[K[3]] = E[3]` $\to$ `X_recovered[1] = S[1]`
   - `X_recovered[K[4]] = E[4]` $\to$ `X_recovered[4] = S[4]`
   - Resultado: `X_recovered == S` exactamente.

---

### Paso 6. Reconstrucción temporal (`irfft`) y generación de archivos de salida
1. **Transformada inversa:**
   ```python
   encoded = irfft(E, n=N)
   ```
   Se pasa explícitamente `n=N` porque $M = \lfloor N/2 \rfloor + 1$ coincide para $N$ par y $N+1$ impar; así se garantiza recuperar exactamente $N = 383\,184$ muestras.
2. **Guardado del audio codificado (`audios/encoded/encoded_{i}.wav`):**
   Se escribe en formato WAV IEEE **`float32`** sin alterar la escala (los cinco audios codificados tienen picos naturales entre `0.2026` y `0.4339`, siempre por debajo de `1.0`). Este formato es directamente compatible con `reverse.py`, que lee el WAV y guarda `recovered.astype(np.float32)`.
3. **Guardado de la clave (`keys/FFT_key_{i}.csv`):**
   ```python
   np.savetxt(ruta_csv, clave_k, fmt="%d", delimiter=",", header="key", comments="")
   ```
   Produce exactamente **una columna de 191 593 enteros** precedida por **una fila de encabezado (`key`)**, coincidiendo con la instrucción original del código Reverse de la profesora en `reverse.py`:
   ```python
   key = np.loadtxt(str(ruta_clave), delimiter=",", skiprows=1, dtype=int)
   ```
4. **Nota sobre la precisión en disco:** Aunque la permutación es matemáticamente exacta, guardar y recargar en WAV `float32` (24 bits de mantisa) introduce un error mínimo de redondeo del orden de $10^{-9}$–$10^{-8}$, lo que produce una recuperación con $\text{SNR} \approx 152.2\text{–}152.4\ \text{dB}$ y correlación de Pearson $\rho = 1.000000000000$, auditivamente idéntica al audio secreto original.
