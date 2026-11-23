"""Algoritmo Imitation: ocultamiento de audio mediante permutación espectral compleja.

Propósito del archivo
---------------------
Este script implementa desde cero la parte **Imitation** del proyecto de
Análisis de Señales (Capítulo 4 — Transformada de Fourier). Toma una señal de
referencia (`target.wav`) y cinco mensajes secretos (`secret1.wav` a `secret5.wav`),
y para cada secreto:

1. Calcula los coeficientes espectrales complejos mediante `rfft`.
2. Identifica los índices permutables `P`, manteniendo fijos DC (k = 0) y,
   cuando la longitud N es par, Nyquist (k = N/2).
3. Ordena de menor a mayor las magnitudes de ambos espectros dentro de `P`.
4. Construye la clave de permutación biyectiva `K` siguiendo el pseudocódigo:
       I1 = argsort(abs(S[P]))
       I2 = argsort(abs(T[P]))
       K = identity_indices
       K[P[I2]] = P[I1]
       E = S[K]
5. Reorganiza los coeficientes complejos completos (parte real e imaginaria,
   conservando conjuntamente magnitud y fase).
6. Reconstruye el audio codificado mediante `irfft(E, n=N)`.
7. Guarda el audio codificado (`audios/encoded/encoded_{i}.wav`) en IEEE
   float32 y la clave (`keys/FFT_key_{i}.csv`) en formato 100 % compatible
   con `reverse.py` (algoritmo original de la profesora).

Dependencias externas
---------------------
- `numpy`: manejo de arreglos, ordenamiento (`argsort`) y exportación CSV.
- `scipy`: lectura/escritura WAV (`scipy.io.wavfile`) y transformadas rápidas
  para señales reales (`scipy.fft.rfft`, `scipy.fft.irfft`).

Uso desde la carpeta raíz del proyecto
--------------------------------------
    python imitation.py
"""

from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
from scipy.fft import irfft, rfft
from scipy.io.wavfile import read as wav_read, write as wav_write


def cargar_audio_mono(ruta_wav: Path) -> tuple[int, np.ndarray]:
    """Carga un archivo WAV y lo convierte a señal mono en `float64` en [-1.0, 1.0).

    Parámetros
    ----------
    ruta_wav : Path
        Ruta del archivo `.wav` que se desea leer.

    Retorna
    -------
    tuple[int, np.ndarray]
        `(fs, senal)` donde `fs` es la frecuencia de muestreo en Hz y `senal`
        es un vector 1D `float64` con las muestras en escala física estándar.

    Razonamiento técnico sobre el escalado PCM
    ------------------------------------------
    Los audios originales del proyecto están grabados en PCM de 24 bits.
    `scipy.io.wavfile.read` entrega las muestras de 24 bits como enteros `int32`
    desplazados 8 bits a la izquierda (rango [-2^31, 2^31 - 256]). Dividir por
    `2**31` equivale exactamente a dividir muestras de 24 bits por `2**23`,
    llevando la señal al rango físico [-1.0, 1.0) sin normalizar arbitrariamente
    por el máximo de cada archivo y conservando las amplitudes relativas reales.
    """
    if not ruta_wav.exists():
        raise FileNotFoundError(f"No se encontró el archivo de audio: {ruta_wav}")

    fs, datos_crudos = wav_read(str(ruta_wav))

    # Conversión exacta según el tipo de dato almacenado en el archivo WAV.
    if datos_crudos.dtype == np.int32:
        # PCM de 24 bits (desplazado a 32 bits por SciPy) o PCM de 32 bits.
        senal = datos_crudos.astype(np.float64) / float(2 ** 31)
    elif datos_crudos.dtype == np.int16:
        # PCM estándar de 16 bits.
        senal = datos_crudos.astype(np.float64) / float(2 ** 15)
    elif datos_crudos.dtype == np.uint8:
        # PCM sin signo de 8 bits centrado en 128.
        senal = (datos_crudos.astype(np.float64) - 128.0) / 128.0
    elif np.issubdtype(datos_crudos.dtype, np.floating):
        # Audio ya almacenado en coma flotante (float32 o float64).
        senal = datos_crudos.astype(np.float64)
    else:
        raise ValueError(f"Formato WAV no soportado ({datos_crudos.dtype}) en {ruta_wav.name}.")

    # Si el audio tuviera múltiples canales (estéreo), se promedian los canales
    # en float64 para trabajar con una única señal monoaural 1D.
    if senal.ndim == 2:
        warnings.warn(
            f"'{ruta_wav.name}' tiene {senal.shape[1]} canales; se convirtió a mono "
            "mediante el promedio aritmético de canales.",
            UserWarning,
            stacklevel=2,
        )
        senal = np.mean(senal, axis=1)
    elif senal.ndim != 1:
        raise ValueError(f"Dimensión inesperada para '{ruta_wav.name}': {senal.shape}")

    return int(fs), senal


def validar_compatibilidad(
    fs_target: int,
    target: np.ndarray,
    fs_secret: int,
    secret: np.ndarray,
    nombre_secret: str,
) -> None:
    """Verifica que Target y Secret tengan exactamente la misma `fs` y longitud `N`.

    Por qué es obligatorio
    ----------------------
    La permutación empareja los coeficientes espectrales índice a índice. Si `fs`
    difiere, un mismo índice `k` corresponde a distintas frecuencias físicas en Hz;
    si `N` difiere, los vectores `rfft` tienen distinta dimensión y no puede
    construirse una permutación biyectiva sin alterar el contenido de las señales.
    """
    if fs_target != fs_secret:
        raise ValueError(
            f"Frecuencia de muestreo incompatible entre target ({fs_target} Hz) "
            f"y {nombre_secret} ({fs_secret} Hz)."
        )
    if len(target) != len(secret):
        raise ValueError(
            f"Longitud incompatible entre target ({len(target)} muestras) "
            f"y {nombre_secret} ({len(secret)} muestras). No se recorta ni rellena."
        )


def obtener_indices_permutables(num_muestras: int, longitud_espectro: int) -> np.ndarray:
    """Construye el vector `P` de posiciones espectrales que pueden permutarse.

    Fundamento matemático
    ---------------------
    Para una señal real de longitud `N`, `rfft` entrega `M = N // 2 + 1` coeficientes:
    - El coeficiente DC (`k = 0`) es siempre estrictamente real: sum(x[n]).
    - Si `N` es PAR, el coeficiente de Nyquist (`k = M - 1 = N/2`) también es
      estrictamente real: sum((-1)^n * x[n]).
    Si se moviera un coeficiente complejo (con parte imaginaria distinta de cero)
    a DC o Nyquist, `irfft` descartaría su parte imaginaria al reconstruir la señal
    real, impidiendo recuperar exactamente el secreto. Por ello se mantienen fijos.
    """
    if num_muestras % 2 == 0:
        # N par: fijamos k=0 (DC) y k=M-1 (Nyquist).
        return np.arange(1, longitud_espectro - 1, dtype=np.int64)
    # N impar: no existe bin exacto de Nyquist; fijamos únicamente k=0 (DC).
    return np.arange(1, longitud_espectro, dtype=np.int64)


def validar_clave_biyectiva(clave_k: np.ndarray, num_muestras: int) -> None:
    """Comprueba que `K` sea una permutación biyectiva válida y respete DC y Nyquist.

    Una permutación biyectiva sobre {0, 1, ..., M - 1} garantiza que cada
    coeficiente complejo de Secret aparezca exactamente una vez en el espectro
    codificado y que `Reverse` pueda invertirla sin ambigüedad ni pérdida.
    """
    m_esperado = num_muestras // 2 + 1
    if len(clave_k) != m_esperado:
        raise ValueError(f"La clave debe tener longitud {m_esperado}, tiene {len(clave_k)}.")

    # Al ordenar K debe coincidir exactamente con [0, 1, ..., M - 1].
    if not np.array_equal(np.sort(clave_k), np.arange(m_esperado, dtype=clave_k.dtype)):
        raise ValueError("La clave K no es una permutación biyectiva válida.")

    if int(clave_k[0]) != 0:
        raise ValueError(f"El índice DC (k=0) debe permanecer fijo; se obtuvo {clave_k[0]}.")

    if num_muestras % 2 == 0 and int(clave_k[-1]) != m_esperado - 1:
        raise ValueError(
            f"El índice Nyquist (k={m_esperado - 1}) debe permanecer fijo para N par; "
            f"se obtuvo {clave_k[-1]}."
        )


def aplicar_imitation(
    target: np.ndarray,
    secret: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Ejecuta el algoritmo Imitation sobre una señal Target y una señal Secret.

    Parámetros
    ----------
    target : np.ndarray
        Señal temporal de referencia (1D, `float64`), cuyo orden de magnitudes
        espectrales sirve como guía para reubicar los coeficientes de Secret.
    secret : np.ndarray
        Señal temporal secreta (1D, `float64`) que se desea ocultar.

    Retorna
    -------
    tuple[np.ndarray, np.ndarray]
        `(encoded, K)` donde:
        - `encoded`: señal temporal real codificada (`float64`, longitud `N`).
        - `K`: clave entera de permutación biyectiva (`int64`, longitud `M`).
    """
    N = len(secret)

    # 1. Aplicamos rFFT porque ambas señales de audio son estrictamente reales.
    #    rFFT aprovecha la simetría conjugada (X[N-k] = X*[k]) y devuelve solo
    #    los M = N // 2 + 1 coeficientes de frecuencias no negativas [0, fs/2].
    T = rfft(target)
    S = rfft(secret)
    M = len(S)

    # 2. Seleccionamos las posiciones permutables P, excluyendo DC (y Nyquist si N es par).
    P = obtener_indices_permutables(num_muestras=N, longitud_espectro=M)

    # 3. Calculamos el orden ascendente de las magnitudes espectrales dentro de P.
    #    Usamos `np.abs(...)` SOLO para establecer el criterio de ordenamiento;
    #    los valores complejos originales (parte real e imaginaria) siguen en S.
    I1 = np.argsort(np.abs(S[P]))
    I2 = np.argsort(np.abs(T[P]))

    # 4. Construimos la clave de permutación K siguiendo exactamente el pseudocódigo:
    #        K = identity_indices
    #        K[P[I2]] = P[I1]
    #    En la posición frecuencial `P[I2[r]]` (donde Target tiene su r-ésima
    #    magnitud ordenada) registramos la posición original `P[I1[r]]` de Secret
    #    (su r-ésima magnitud ordenada).
    K = np.arange(M, dtype=np.int64)
    K[P[I2]] = P[I1]

    # Verificamos matemáticamente que K sea biyectiva y conserve fijos DC y Nyquist.
    validar_clave_biyectiva(K, num_muestras=N)

    # 5. Aplicamos la permutación sobre los COEFICIENTES COMPLEJOS COMPLETOS de Secret.
    #    Cada elemento E[m] = S[K[m]] conserva intactas tanto su magnitud como su fase.
    E = S[K]

    # 6. Volvemos al dominio temporal mediante la transformada inversa irfft.
    #    Pasamos explícitamente `n=N` para garantizar la longitud exacta original.
    encoded = irfft(E, n=N)

    return encoded, K


def guardar_clave_csv(clave_k: np.ndarray, ruta_csv: Path) -> None:
    """Guarda la clave `K` en CSV compatible con `Reverse_FFT_GitHub.ipynb`.

    El código Reverse original de la profesora carga la clave con:
        key = np.loadtxt('FFT_key.csv', delimiter=',', skiprows=1, dtype=int)
    Por lo tanto, el archivo debe tener exactamente UNA columna de enteros con
    UNA fila de encabezado (`key`) y sin prefijo de comentario (`comments=""`).
    """
    ruta_csv.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        str(ruta_csv),
        np.asarray(clave_k, dtype=np.int64),
        fmt="%d",
        delimiter=",",
        header="key",
        comments="",
    )


def guardar_audio_codificado(ruta_wav: Path, encoded: np.ndarray, fs: int) -> None:
    """Guarda el audio codificado en WAV IEEE `float32` sin alterar su escala.

    El notebook `Reverse_FFT_GitHub.ipynb` lee `encoded.wav` directamente con
    `scipy.io.wavfile.read` y guarda la recuperación como `recovered.astype(np.float32)`.
    Guardar `encoded` en `float32` en el rango físico [-1.0, 1.0] garantiza
    compatibilidad exacta de escala y mínima pérdida de precisión (~152 dB de SNR).
    """
    ruta_wav.parent.mkdir(parents=True, exist_ok=True)
    pico = float(np.max(np.abs(encoded)))
    if pico > 1.0:
        warnings.warn(
            f"La señal '{ruta_wav.name}' supera 1.0 (pico={pico:.4f}). "
            "No se aplica recorte para no destruir la reversibilidad espectral.",
            UserWarning,
            stacklevel=2,
        )
    wav_write(str(ruta_wav), fs, encoded.astype(np.float32))


def main(base_dir: str | Path | None = None) -> None:
    """Procesa los cinco audios secretos usando el mismo Target.

    Compatible tanto con ejecución local (`python imitation.py`) como con
    Google Colab (`!python imitation.py`, `%run imitation.py` o `import imitation`).
    """
    raiz = (
        Path(base_dir).resolve()
        if base_dir is not None
        else (Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd())
    )
    dir_originales = raiz / "audios" / "originales"
    dir_codificados = raiz / "audios" / "encoded"
    dir_claves = raiz / "keys"

    # Cargamos el audio de referencia (Target) una única vez.
    fs_target, target = cargar_audio_mono(dir_originales / "target.wav")
    print(
        f"[Target] target.wav cargado: fs={fs_target} Hz, N={len(target)} muestras, "
        f"duración={len(target) / fs_target:.4f} s"
    )

    # Repetimos el proceso Imitation para secret1.wav hasta secret5.wav sin sobrescribir.
    for i in range(1, 6):
        nombre_secret = f"secret{i}.wav"
        fs_secret, secret = cargar_audio_mono(dir_originales / nombre_secret)

        # Validamos que Target y Secret sean estrictamente compatibles.
        validar_compatibilidad(fs_target, target, fs_secret, secret, nombre_secret)

        # Ejecutamos la permutación espectral compleja.
        encoded, clave_k = aplicar_imitation(target, secret)

        # Guardamos audio codificado y clave con numeración consistente (1 a 5).
        ruta_encoded = dir_codificados / f"encoded_{i}.wav"
        ruta_clave = dir_claves / f"FFT_key_{i}.csv"

        guardar_audio_codificado(ruta_encoded, encoded, fs_secret)
        guardar_clave_csv(clave_k, ruta_clave)

        print(
            f"[Imitation {i}/5] {nombre_secret} -> {ruta_encoded.name} "
            f"(pico={np.max(np.abs(encoded)):.4f}) | clave -> {ruta_clave.name} "
            f"(M={len(clave_k)} índices)"
        )

    print("\nProceso Imitation completado para los 5 audios secretos.")


if __name__ == "__main__":
    main()
