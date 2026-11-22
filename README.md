# Proyecto de Análisis de Señales — Ocultamiento y Recuperación Espectral (*Imitation* y *Reverse*)

Proyecto en Python que oculta cinco mensajes de audio mediante permutación de sus coeficientes espectrales de Fourier (**Imitation**) y los recupera de manera independiente aplicando la permutación inversa (**Reverse**).

---

## 1. Flujo de Ejecución

```text
audios/originales/                  audios/encoded/ + keys/                 audios/recovered/
(target.wav + secret1..5.wav)       (encoded_1..5.wav + FFT_key_1..5.csv)   (recovered_1..5.wav)
             │                                        │                               ▲
             ▼                                        ▼                               │
     [ imitation.py ] ──────────────────────►  [ reverse.py ] ────────────────────────┘
      (PASO 1: Codifica)                        (PASO 2: Recupera)
```

---

## 2. Estructura del Proyecto

```text
Proyecto_Analisis_Senales/
│
├── audios/
│   ├── originales/                # Audios originales: target.wav y secret1.wav .. secret5.wav
│   ├── encoded/                   # Audios codificados: encoded_1.wav .. encoded_5.wav
│   └── recovered/                 # Audios recuperados: recovered_1.wav .. recovered_5.wav
│
├── keys/
│   ├── FFT_key_1.csv              # Claves de permutación generadas por imitation.py
│   ├── FFT_key_2.csv
│   ├── FFT_key_3.csv
│   ├── FFT_key_4.csv
│   └── FFT_key_5.csv
│
├── imitation.py                   # Programa 1: codificación espectral (genera encoded + claves)
├── reverse.py                     # Programa 2: recuperación espectral (genera recovered)
├── README.md                      # Guía de instalación y ejecución
└── EXPLICACION_IMITATION.md       # Explicación matemática y paso a paso de imitation.py
```

---

## 3. Requisitos e Instalación de Dependencias

El proyecto requiere **Python 3.10 o superior** y únicamente las dos librerías externas utilizadas por `imitation.py` y `reverse.py`:

- `numpy`
- `scipy`

### Instalación en Windows (PowerShell o CMD)

Abre una terminal en la carpeta `Proyecto_Analisis_Senales` y ejecuta:

```powershell
python --version
pip install numpy scipy
```

---

## 4. Orden de Ejecución Obligatorio

Este proyecto está compuesto por **dos programas independientes** que deben ejecutarse desde la terminal **estrictamente en el siguiente orden**, ubicados dentro de la carpeta `Proyecto_Analisis_Senales`:

### PASO 1. Imitation — Codificación de los audios

- **Archivo:** `imitation.py`
- **Comando (PowerShell / terminal):**

```powershell
python imitation.py
```

**Qué realiza este programa:**

- Carga el audio de referencia (`audios/originales/target.wav`) y los cinco audios secretos originales (`audios/originales/secret1.wav` a `secret5.wav`).
- Calcula sus transformadas rápidas de Fourier para señales reales (`rfft`).
- Realiza la permutación espectral de los coeficientes complejos completos (manteniendo fijos DC y Nyquist).
- Genera los cinco audios codificados (`encoded_1.wav` a `encoded_5.wav`).
- Genera las cinco claves de permutación en formato CSV (`FFT_key_1.csv` a `FFT_key_5.csv`).
- Guarda los resultados en `audios/encoded/` y `keys/`, respectivamente.

> **Obligatorio:** Este paso **debe ejecutarse primero**. Los audios codificados y las claves CSV que produce son indispensables para poder ejecutar el siguiente programa.

---

### PASO 2. Reverse — Recuperación de los audios

- **Archivo:** `reverse.py`
- **Comando (PowerShell / terminal):**

```powershell
python reverse.py
```

**Qué realiza este programa:**

- Carga los cinco audios codificados desde `audios/encoded/`.
- Lee las cinco claves CSV correspondientes desde `keys/` (generadas previamente por `imitation.py`).
- Aplica el procedimiento inverso de recuperación sobre los coeficientes complejos de Fourier (`X_recovered[key] = X_encoded`).
- Reconstruye los cinco audios originales mediante la transformada inversa (`irfft`).
- Guarda los cinco audios recuperados (`recovered_1.wav` a `recovered_5.wav`) en `audios/recovered/`.

> **Importante:** `reverse.py` **no genera claves nuevas** ni modifica las existentes; depende estrictamente de los archivos producidos por `imitation.py`.

---

## 5. Correspondencia entre Archivos

Cada audio secreto mantiene una correspondencia exacta uno a uno entre sus archivos:

| Audio original | Audio codificado (`PASO 1`) | Clave CSV (`PASO 1`) | Audio recuperado (`PASO 2`) |
|---|---|---|---|
| `audios/originales/secret1.wav` | `audios/encoded/encoded_1.wav` | `keys/FFT_key_1.csv` | `audios/recovered/recovered_1.wav` |
| `audios/originales/secret2.wav` | `audios/encoded/encoded_2.wav` | `keys/FFT_key_2.csv` | `audios/recovered/recovered_2.wav` |
| `audios/originales/secret3.wav` | `audios/encoded/encoded_3.wav` | `keys/FFT_key_3.csv` | `audios/recovered/recovered_3.wav` |
| `audios/originales/secret4.wav` | `audios/encoded/encoded_4.wav` | `keys/FFT_key_4.csv` | `audios/recovered/recovered_4.wav` |
| `audios/originales/secret5.wav` | `audios/encoded/encoded_5.wav` | `keys/FFT_key_5.csv` | `audios/recovered/recovered_5.wav` |

---

## 6. Cómo Regenerar los Archivos desde Cero

Si deseas reconstruir todos los archivos generados desde cero (por ejemplo, si se eliminaron los audios codificados, las claves o los audios recuperados), sigue exactamente estos pasos en PowerShell:

1. **Conserva intactos los audios originales** en `audios/originales/` (`target.wav` y `secret1.wav`–`secret5.wav`).
2. **Ejecuta Imitation:**
   ```powershell
   python imitation.py
   ```
3. **Verifica** que se hayan generado los cinco archivos en `audios/encoded/` (`encoded_1.wav`–`encoded_5.wav`) y las cinco claves en `keys/` (`FFT_key_1.csv`–`FFT_key_5.csv`):
   ```powershell
   Get-ChildItem audios\encoded, keys
   ```
4. **Ejecuta Reverse:**
   ```powershell
   python reverse.py
   ```
5. **Verifica** que se hayan generado los cinco audios recuperados en `audios/recovered/` (`recovered_1.wav`–`recovered_5.wav`):
   ```powershell
   Get-ChildItem audios\recovered
   ```

> Si se eliminan las claves en `keys/` o los audios codificados en `audios/encoded/`, **deben regenerarse ejecutando `python imitation.py` antes de volver a ejecutar `python reverse.py`**.

---

## 7. Advertencias Importantes

- **No ejecutes `reverse.py` antes de `imitation.py`:** sin los audios codificados ni sus claves CSV, la recuperación fallará.
- **No intercambies las claves entre audios:** aplicar la clave de un secreto sobre otro audio codificado impide recuperar el mensaje (producirá ruido ininteligible).
- **No modifiques manualmente los archivos CSV de `keys/`:** alterar el encabezado, el orden o algún índice rompe la biyectividad de la permutación.
- **No mezcles claves de ejecuciones diferentes con audios codificados anteriores:** cada par `encoded_i.wav` y `FFT_key_i.csv` debe pertenecer a la misma ejecución de `imitation.py`.
- **No elimines los audios originales** en `audios/originales/`: son la fuente necesaria para regenerar todo el proceso.
- **Mantén siempre la correspondencia numérica** (`1` a `5`) entre `encoded_i.wav`, `FFT_key_i.csv` y `recovered_i.wav`.

---

## 8. Documentación Complementaria

Para una explicación detallada de las variables, operaciones matemáticas (`rfft`, `argsort`, permutación biyectiva `K`, DC/Nyquist e `irfft`) y generación de claves de `imitation.py`, consulta [EXPLICACION_IMITATION.md](EXPLICACION_IMITATION.md).
