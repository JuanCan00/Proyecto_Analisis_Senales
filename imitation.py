# Importaciones
from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
from scipy.fft import irfft, rfft
from scipy.io.wavfile import read as wav_read, write as wav_write


# Carga
def cargar_audio_mono(ruta_wav: Path) -> tuple[int, np.ndarray]:
    if not ruta_wav.exists():
        raise FileNotFoundError(f"No se encontró el archivo de audio: {ruta_wav}")

    fs, datos_crudos = wav_read(str(ruta_wav))

    if datos_crudos.dtype == np.int32:
        senal = datos_crudos.astype(np.float64) / float(2 ** 31)
    elif datos_crudos.dtype == np.int16:
        senal = datos_crudos.astype(np.float64) / float(2 ** 15)
    elif datos_crudos.dtype == np.uint8:
        senal = (datos_crudos.astype(np.float64) - 128.0) / 128.0
    elif np.issubdtype(datos_crudos.dtype, np.floating):
        senal = datos_crudos.astype(np.float64)
    else:
        raise ValueError(f"Formato WAV no soportado ({datos_crudos.dtype}) en {ruta_wav.name}.")

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


# Compatibilidad
def validar_compatibilidad(
    fs_target: int,
    target: np.ndarray,
    fs_secret: int,
    secret: np.ndarray,
    nombre_secret: str,
) -> None:
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


# Permutables
def obtener_indices_permutables(num_muestras: int, longitud_espectro: int) -> np.ndarray:
    if num_muestras % 2 == 0:
        return np.arange(1, longitud_espectro - 1, dtype=np.int64)
    return np.arange(1, longitud_espectro, dtype=np.int64)


# Validacion
def validar_clave_biyectiva(clave_k: np.ndarray, num_muestras: int) -> None:
    m_esperado = num_muestras // 2 + 1
    if len(clave_k) != m_esperado:
        raise ValueError(f"La clave debe tener longitud {m_esperado}, tiene {len(clave_k)}.")

    if not np.array_equal(np.sort(clave_k), np.arange(m_esperado, dtype=clave_k.dtype)):
        raise ValueError("La clave K no es una permutación biyectiva válida.")

    if int(clave_k[0]) != 0:
        raise ValueError(f"El índice DC (k=0) debe permanecer fijo; se obtuvo {clave_k[0]}.")

    if num_muestras % 2 == 0 and int(clave_k[-1]) != m_esperado - 1:
        raise ValueError(
            f"El índice Nyquist (k={m_esperado - 1}) debe permanecer fijo para N par; "
            f"se obtuvo {clave_k[-1]}."
        )


# Imitation
def aplicar_imitation(
    target: np.ndarray,
    secret: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    N = len(secret)

    # FFT
    T = rfft(target)
    S = rfft(secret)
    M = len(S)

    # Indices
    P = obtener_indices_permutables(num_muestras=N, longitud_espectro=M)

    # Ranking
    I1 = np.argsort(np.abs(S[P]))
    I2 = np.argsort(np.abs(T[P]))

    # Clave
    K = np.arange(M, dtype=np.int64)
    K[P[I2]] = P[I1]
    validar_clave_biyectiva(K, num_muestras=N)

    # Permutacion
    E = S[K]

    # Reconstruccion
    encoded = irfft(E, n=N)

    return encoded, K


# Exportacion
def guardar_clave_csv(clave_k: np.ndarray, ruta_csv: Path) -> None:
    ruta_csv.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        str(ruta_csv),
        np.asarray(clave_k, dtype=np.int64),
        fmt="%d",
        delimiter=",",
        header="key",
        comments="",
    )


# Codificado
def guardar_audio_codificado(ruta_wav: Path, encoded: np.ndarray, fs: int) -> None:
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


# Principal
def main(base_dir: str | Path | None = None) -> None:
    raiz = (
        Path(base_dir).resolve()
        if base_dir is not None
        else (Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd())
    )
    dir_originales = raiz / "audios" / "originales"
    dir_codificados = raiz / "audios" / "encoded"
    dir_claves = raiz / "keys"

    fs_target, target = cargar_audio_mono(dir_originales / "target.wav")
    print(
        f"[Target] target.wav cargado: fs={fs_target} Hz, N={len(target)} muestras, "
        f"duración={len(target) / fs_target:.4f} s"
    )

    for i in range(1, 6):
        nombre_secret = f"secret{i}.wav"
        fs_secret, secret = cargar_audio_mono(dir_originales / nombre_secret)

        validar_compatibilidad(fs_target, target, fs_secret, secret, nombre_secret)

        encoded, clave_k = aplicar_imitation(target, secret)

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


# Arranque
if __name__ == "__main__":
    main()
