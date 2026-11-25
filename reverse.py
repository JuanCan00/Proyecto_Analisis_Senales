# Importaciones
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.fft import irfft, rfft
from scipy.io.wavfile import read, write


# Validacion
def verificar_clave_compatible(key: np.ndarray, num_muestras: int, longitud_espectro: int) -> None:
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


# Reverse
def recuperar_audio(
    ruta_encoded: Path,
    ruta_clave: Path,
    ruta_recovered: Path,
) -> tuple[int, np.ndarray]:
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

    # Carga
    fs, encoded = read(str(ruta_encoded))
    key = np.loadtxt(str(ruta_clave), delimiter=",", skiprows=1, dtype=int)

    # FFT
    X_encoded = rfft(encoded.astype(float))
    verificar_clave_compatible(key, num_muestras=len(encoded), longitud_espectro=len(X_encoded))

    # Despermutacion
    X_recovered = np.empty_like(X_encoded)
    X_recovered[key] = X_encoded

    # Reconstruccion
    recovered = irfft(X_recovered, n=len(encoded))

    # Guardado
    ruta_recovered.parent.mkdir(parents=True, exist_ok=True)
    write(str(ruta_recovered), fs, recovered.astype(np.float32))

    return int(fs), recovered


# Principal
def main(base_dir: str | Path | None = None) -> None:
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


# Arranque
if __name__ == "__main__":
    main()
