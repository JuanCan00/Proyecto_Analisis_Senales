# Proyecto de Análisis de Señales — Ocultamiento y Recuperación Espectral (*Imitation* y *Reverse*)

Proyecto en Python que oculta cinco mensajes de audio mediante la permutación biyectiva de sus coeficientes espectrales de Fourier (**Imitation**) y los recupera de manera independiente aplicando la permutación inversa (**Reverse**).

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-FFT%20%26%20Arrays-013243?style=flat-square&logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-rFFT%20%26%20WAV-8CAAE6?style=flat-square&logo=scipy&logoColor=white)
![Área](https://img.shields.io/badge/%C3%81rea-An%C3%A1lisis%20de%20Se%C3%B1ales-0F766E?style=flat-square)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JuanCan00/Proyecto_Analisis_Senales/blob/main/Ejecutar_en_Colab.ipynb)

---

## Descripción

En el dominio de la frecuencia, cada señal de audio real se representa mediante sus coeficientes complejos de Fourier obtenidos con `rfft`. Este proyecto implementa un flujo completo en dos etapas independientes:

1. **Codificación espectral (`imitation.py`)**: reorganiza los coeficientes complejos completos de cada audio secreto (`secret1.wav`–`secret5.wav`) para imitar el orden de magnitudes espectrales de un audio de referencia (`target.wav`), manteniendo fijos los componentes DC ($k = 0$) y Nyquist ($k = N/2$). Como resultado, el audio codificado suena como ruido ininteligible mientras conserva intacta toda la información espectral original.
2. **Recuperación espectral (`reverse.py`)**: lee cada audio codificado junto con su clave de permutación CSV (`FFT_key_i.csv`), aplica la despermutación inversa (`X_recovered[key] = X_encoded`) y reconstruye el mensaje original con `irfft`.

---

## Flujo de ejecución

```text
audios/originales/                  audios/encoded/ + keys/                 audios/recovered/
(target.wav + secret1..5.wav)       (encoded_1..5.wav + FFT_key_1..5.csv)   (recovered_1..5.wav)
             │                                        │                               ▲
             ▼                                        ▼                               │
     [ imitation.py ] ──────────────────────►  [ reverse.py ] ────────────────────────┘
      (PASO 1: Codifica)                        (PASO 2: Recupera)
```

---

## Tecnologías

| Componente | Uso |
| :--- | :--- |
| **Python 3.10+** | Ejecución de ambos scripts independientes |
| **NumPy** | Procesamiento vectorial, ordenamiento (`argsort`) y lectura/escritura CSV |
| **SciPy** | Lectura/escritura de archivos WAV (`scipy.io.wavfile`) y transformadas `rfft` / `irfft` (`scipy.fft`) |

---

## Estructura del proyecto

```text
Proyecto_Analisis_Senales/
├── audios/
│   ├── originales/                # Audios fuente: target.wav y secret1.wav .. secret5.wav
│   ├── encoded/                   # Audios codificados: encoded_1.wav .. encoded_5.wav
│   └── recovered/                 # Audios recuperados: recovered_1.wav .. recovered_5.wav
├── keys/
│   ├── FFT_key_1.csv              # Claves de permutación generadas por imitation.py
│   ├── FFT_key_2.csv
│   ├── FFT_key_3.csv
│   ├── FFT_key_4.csv
│   └── FFT_key_5.csv
├── imitation.py                   # Paso 1: codificación espectral (genera encoded + claves)
├── reverse.py                     # Paso 2: recuperación espectral (genera recovered)
├── Ejecutar_en_Colab.ipynb        # Cuaderno lanzador para ejecutar en Google Colab
├── EXPLICACION_IMITATION.md       # Desarrollo matemático y computacional de imitation.py
└── README.md                      # Documentación principal del repositorio
```

---

## Requisitos e instalación

El proyecto requiere **Python 3.10 o superior** y únicamente las librerías `numpy` y `scipy`.

```powershell
git clone https://github.com/JuanCan00/Proyecto_Analisis_Senales.git
cd Proyecto_Analisis_Senales
python --version
pip install numpy scipy
```

---

## Orden de ejecución obligatorio

Los dos programas son independientes y deben ejecutarse desde la carpeta raíz del repositorio **estrictamente en el siguiente orden**:

### Paso 1. Imitation — Codificación de los audios

```powershell
python imitation.py
```

**Qué realiza `imitation.py`:**
- Carga el audio de referencia (`audios/originales/target.wav`) y los cinco audios secretos (`secret1.wav` a `secret5.wav`) convirtiéndolos con escalado PCM exacto a `float64` monoaural.
- Calcula sus espectros complejos mediante `rfft`.
- Construye la clave de permutación biyectiva `K` manteniendo fijos DC y Nyquist, y reubica los coeficientes complejos completos (`E = S[K]`).
- Genera los audios codificados en `audios/encoded/encoded_1..5.wav` (IEEE `float32`) y las claves en `keys/FFT_key_1..5.csv`.

> [!IMPORTANT]
> Este paso **debe ejecutarse primero**. Los audios codificados y las claves CSV que produce son indispensables para ejecutar `reverse.py`.

### Paso 2. Reverse — Recuperación de los audios

```powershell
python reverse.py
```

**Qué realiza `reverse.py`:**
- Carga los cinco audios codificados desde `audios/encoded/`.
- Valida y lee las cinco claves CSV correspondientes desde `keys/`.
- Aplica la despermutación inversa sobre los coeficientes complejos de Fourier (`X_recovered[key] = X_encoded`).
- Reconstruye los mensajes con `irfft` y los guarda en `audios/recovered/recovered_1..5.wav`.

> [!NOTE]
> `reverse.py` **no genera ni modifica claves**: depende estrictamente de los archivos generados por `imitation.py`.

---

## Ejecución en Google Colab

Los scripts `imitation.py` y `reverse.py` son multiplataforma (funcionan sin cambios en **Windows/Linux local** y en **Google Colab**). Como el repositorio es **privado**, puedes ejecutarlo en Colab de dos formas:

### Opción A — Abrir `Ejecutar_en_Colab.ipynb` directamente desde Colab (recomendado)

1. Abre [Google Colab](https://colab.research.google.com/).
2. Ve a **Archivo → Abrir cuaderno → GitHub**.
3. Marca la casilla **Incluir repositorios privados** y autoriza tu cuenta de GitHub.
4. Selecciona `JuanCan00/Proyecto_Analisis_Senales` y abre **`Ejecutar_en_Colab.ipynb`**.
5. Ejecuta las celdas en orden:
   - La **celda 1** clona el repositorio en Colab (solicita de forma segura tu *GitHub Personal Access Token* con permiso `repo`, o lo lee automáticamente desde *Colab Secrets* si lo guardaste como `GITHUB_TOKEN`).
   - Las **celdas 3 y 4** ejecutan `!python imitation.py` y `!python reverse.py`.
   - La **celda 5** permite escuchar los audios originales, codificados y recuperados directamente en el navegador.

### Opción B — Clonar desde un cuaderno en blanco de Colab

```python
from getpass import getpass

token = getpass("GitHub Personal Access Token (permiso repo): ")
!git clone https://{token}@github.com/JuanCan00/Proyecto_Analisis_Senales.git
%cd Proyecto_Analisis_Senales
!python imitation.py
!python reverse.py
```

*(Si en algún momento cambias el repositorio a público, el botón **Open in Colab** de la parte superior funcionará directamente y no necesitarás token.)*

---

## Correspondencia entre archivos

| Audio original | Audio codificado (`Paso 1`) | Clave CSV (`Paso 1`) | Audio recuperado (`Paso 2`) |
| :--- | :--- | :--- | :--- |
| `audios/originales/secret1.wav` | `audios/encoded/encoded_1.wav` | `keys/FFT_key_1.csv` | `audios/recovered/recovered_1.wav` |
| `audios/originales/secret2.wav` | `audios/encoded/encoded_2.wav` | `keys/FFT_key_2.csv` | `audios/recovered/recovered_2.wav` |
| `audios/originales/secret3.wav` | `audios/encoded/encoded_3.wav` | `keys/FFT_key_3.csv` | `audios/recovered/recovered_3.wav` |
| `audios/originales/secret4.wav` | `audios/encoded/encoded_4.wav` | `keys/FFT_key_4.csv` | `audios/recovered/recovered_4.wav` |
| `audios/originales/secret5.wav` | `audios/encoded/encoded_5.wav` | `keys/FFT_key_5.csv` | `audios/recovered/recovered_5.wav` |

---

## Regenerar los archivos desde cero

1. Conserva intactos los audios originales en `audios/originales/` (`target.wav` y `secret1.wav`–`secret5.wav`).
2. Ejecuta la codificación y verifica la salida:
   ```powershell
   python imitation.py
   Get-ChildItem audios\encoded, keys
   ```
3. Ejecuta la recuperación y verifica la salida:
   ```powershell
   python reverse.py
   Get-ChildItem audios\recovered
   ```

---

## Advertencias importantes

- **No ejecutes `reverse.py` antes de `imitation.py`:** sin los audios codificados ni sus claves CSV, la recuperación fallará.
- **No intercambies claves entre audios:** aplicar la clave de un mensaje sobre otro audio codificado produce ruido ininteligible.
- **No edites manualmente los archivos en `keys/`:** alterar el encabezado, el orden o algún índice rompe la biyectividad de la permutación.
- **No mezcles claves de ejecuciones diferentes** con audios codificados anteriores.
- **Conserva siempre `audios/originales/`** y la correspondencia numérica (`1` a `5`) entre `encoded_i.wav`, `FFT_key_i.csv` y `recovered_i.wav`.

---

## Documentación complementaria

Para el desarrollo matemático y computacional detallado de `imitation.py` (variables, escalado PCM de 24 bits, `rfft`, `argsort`, permutación biyectiva `K`, DC/Nyquist e `irfft`), consulta [EXPLICACION_IMITATION.md](EXPLICACION_IMITATION.md).

---

## Autor

**Juan Pablo Cano** · [@JuanCan00](https://github.com/JuanCan00)  
Ingeniería en Telecomunicaciones — Universidad Militar Nueva Granada
