"""Algoritmo Reverse: recuperación independiente de los cinco mensajes secretos.

Origen y fidelidad al código de la profesora
--------------------------------------------
Este archivo extrae de manera fiel el algoritmo **Reverse** suministrado por la
profesora Dora María Ballesteros en el notebook original `Reverse_FFT_GitHub.ipynb`
(celda de código principal), conservando exactamente sus operaciones matemáticas,
la carga de la clave CSV, la convención de despermutación y el guardado en WAV:

    fs, encoded = read('encoded.wav')
    key = np.loadtxt('FFT_key.csv', delimiter=',', skiprows=1, dtype=int)
    X_encoded = rfft(encoded.astype(float))
    X_recovered = np.empty_like(X_encoded)
    X_recovered[key] = X_encoded
    recovered = irfft(X_recovered, n=len(encoded))
    write('recovered.wav', fs, recovered.astype(np.float32))

Adaptaciones estrictamente necesarias para ejecución independiente en terminal
------------------------------------------------------------------------------
1. Se retiró `from IPython.display import Audio, display` y `display(Audio(...))`,
   ya que corresponden al reproductor interactivo exclusivo de Jupyter Notebook.
2. Se encapsuló el procedimiento original en la función `recuperar_audio` y se
   agregó `main()` para procesar automáticamente los cinco pares ya existentes
   (`audios/encoded/encoded_1.wav`..`encoded_5.wav` y `keys/FFT_key_1.csv`..`FFT_key_5.csv`)
   generando `audios/recovered/recovered_1.wav`..`recovered_5.wav`.
3. Este programa **NO genera ni modifica claves CSV**: únicamente lee las claves
   producidas previamente por `imitation.py`.

Dependencias externas
---------------------
- `numpy`: lectura de las claves CSV (`np.loadtxt`) y despermutación indexada.
- `scipy`: lectura/escritura WAV (`scipy.io.wavfile.read`, `write`) y transformadas
  rápidas para señales reales (`scipy.fft.rfft`, `irfft`).

Uso independiente desde la terminal
-----------------------------------
    python reverse.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.fft import irfft, rfft
from scipy.io.wavfile import read, write


def verificar_clave_compatible(key: np.ndarray, num_muestras: int, longitud_espectro: int) -> None:
    """Comprueba (en modo solo lectura) que la clave CSV sea válida antes de despermutar.

    Parámetros
    ----------
    key : np.ndarray
        Vector 1D de enteros cargado desde `keys/FFT_key_{i}.csv`.
    num_muestras : int
        Número total de muestras `N` del audio codificado.
    longitud_espectro : int
        Número de coeficientes complejos `M = len(X_encoded)` devueltos por `rfft`.

    Qué valida
    ----------
    - Que `key` tenga exactamente longitud `M = N // 2 + 1`.
    - Que sea una permutación biyectiva de `{0, 1, ..., M - 1}` (sin repetidos).
    - Que mantenga fijos DC (`key[0] == 0`) y, si `N` es par, Nyquist (`key[M - 1] == M - 1`).
    """
    if key.ndim != 1 or len(key) != longitud_espectro:
        raise ValueError(
            f"La clave tiene forma {key.shape}, pero el espectro rFFT tiene longitud "
            f"{longitud_espectro}."
        )
    if not np.array_equal(np.sort(key), np.arange(longitud_espectro, dtype=key.dtype)):
        raise ValueError("La clave CSV no es una permutación biyectiva válida de {0..M-1}.")
    if int(key[0]) != 0:
        raise ValueError(f"El coeficiente DC (k=0) no está fijo en la clave: key[0]={key[0]}.")
    if num_muestras % 2 == 0 and int(key[-1]) != longitud_espectro - 1:
        raise ValueError(
            f"Para N par ({num_muestras}), el coeficiente Nyquist debe estar fijo "
            f"(key[-1]={longitud_espectro - 1}), se encontró {key[-1]}."
        )


def recuperar_audio(
    ruta_encoded: Path,
    ruta_clave: Path,
    ruta_recovered: Path,
) -> tuple[int, np.ndarray]:
    """Recupera un mensaje secreto aplicando el algoritmo Reverse original de la profesora.

    Parámetros
    ----------
    ruta_encoded : Path
        Ruta del audio codificado existente (p. ej., `audios/encoded/encoded_1.wav`).
    ruta_clave : Path
        Ruta de la clave de permutación existente (p. ej., `keys/FFT_key_1.csv`).
    ruta_recovered : Path
        Ruta de salida donde se guardará el audio recuperado
        (p. ej., `audios/recovered/recovered_1.wav`).

    Retorna
    -------
    tuple[int, np.ndarray]
        `(fs, recovered)` con la frecuencia de muestreo (Hz) y la señal temporal
        recuperada en doble precisión (guardada en disco como IEEE `float32`).
    """
    if not ruta_encoded.exists():
        raise FileNotFoundError(
            f"No se encontró el audio codificado '{ruta_encoded}'. "
            "Ejecute primero 'python imitation.py'."
        )
    if not ruta_clave.exists():
        raise FileNotFoundError(
            f"No se encontró la clave '{ruta_clave}'. "
            "Ejecute primero 'python imitation.py'."
        )

    # 1. CARGAR AUDIO CODIFICADO Y CLAVE (instrucciones originales de la profesora)
    #    `read` devuelve la frecuencia de muestreo `fs` y el arreglo `encoded`.
    #    `np.loadtxt(..., skiprows=1, dtype=int)` omite la fila de encabezado (`key`)
    #    y carga los M = 191 593 enteros de la permutación generada por imitation.py.
    fs, encoded = read(str(ruta_encoded))
    key = np.loadtxt(str(ruta_clave), delimiter=",", skiprows=1, dtype=int)

    # 2. TRANSFORMADA DE FOURIER (rFFT de señales reales)
    #    Convertimos las muestras a punto flotante (`float64`) y calculamos el
    #    espectro complejo `X_encoded` de frecuencias no negativas [0, fs/2].
    X_encoded = rfft(encoded.astype(float))

    # Verificación preventiva de integridad de la clave (sin modificar el archivo CSV).
    verificar_clave_compatible(key, num_muestras=len(encoded), longitud_espectro=len(X_encoded))

    # 3. DESPERMUTAR COEFICIENTES COMPLEJOS (permutación inversa exacta)
    #    En imitation.py se construyó el espectro codificado mediante:
    #        E[m] = S[K[m]]   para m = 0, 1, ..., M - 1
    #    Es decir, el coeficiente almacenado en la posición `m` de `X_encoded`
    #    proviene originalmente de la frecuencia `key[m]` del mensaje secreto `S`.
    #    Al ejecutar `X_recovered[key] = X_encoded`, Python realiza la asignación
    #    inversa:
    #        X_recovered[key[m]] = X_encoded[m] = S[key[m]]
    #    devolviendo cada coeficiente complejo completo (parte real e imaginaria,
    #    es decir magnitud y fase) a su posición frecuencial original.
    X_recovered = np.empty_like(X_encoded)
    X_recovered[key] = X_encoded

    # 4. TRANSFORMADA INVERSA (irFFT al dominio temporal)
    #    Reconstruye la señal real en el tiempo indicando explícitamente `n=len(encoded)`
    #    para preservar exactamente el número original de muestras N = 383 184.
    recovered = irfft(X_recovered, n=len(encoded))

    # 5. GUARDAR AUDIO RECUPERADO EN WAV IEEE float32 (instrucción original)
    ruta_recovered.parent.mkdir(parents=True, exist_ok=True)
    write(str(ruta_recovered), fs, recovered.astype(np.float32))

    return int(fs), recovered


def main(base_dir: str | Path | None = None) -> None:
    """Ejecuta Reverse de forma independiente para los cinco audios codificados.

    Compatible tanto con ejecución local (`python reverse.py`) como con
    Google Colab (`!python reverse.py`, `%run reverse.py` o `import reverse`).
    """
    raiz = (
        Path(base_dir).resolve()
        if base_dir is not None
        else (Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd())
    )
    dir_encoded = raiz / "audios" / "encoded"
    dir_keys = raiz / "keys"
    dir_recovered = raiz / "audios" / "recovered"

    for i in range(1, 6):
        ruta_encoded = dir_encoded / f"encoded_{i}.wav"
        ruta_clave = dir_keys / f"FFT_key_{i}.csv"
        ruta_recovered = dir_recovered / f"recovered_{i}.wav"

        fs, recovered = recuperar_audio(
            ruta_encoded=ruta_encoded,
            ruta_clave=ruta_clave,
            ruta_recovered=ruta_recovered,
        )
        print(
            f"[Reverse {i}/5] {ruta_encoded.name} + {ruta_clave.name} -> "
            f"{ruta_recovered.name} (fs={fs} Hz, N={len(recovered)}, "
            f"rango=[{recovered.min():.4f}, {recovered.max():.4f}])"
        )

    print("\nProceso Reverse completado para los 5 audios recuperados.")


if __name__ == "__main__":
    main()
