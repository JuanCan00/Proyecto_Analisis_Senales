# Importaciones
from __future__ import annotations

from pathlib import Path

import numpy as np

from imitation import cargar_audio_mono


# Metricas
def calcular_snr_db(original: np.ndarray, recuperada: np.ndarray) -> float:
    energia_senal = float(np.sum(original ** 2))
    energia_error = float(np.sum((original - recuperada) ** 2))
    if energia_error == 0.0:
        return float("inf")
    return float(10.0 * np.log10(energia_senal / energia_error))


# Correlacion
def calcular_pearson(original: np.ndarray, recuperada: np.ndarray) -> float:
    x_centrado = original - np.mean(original)
    y_centrado = recuperada - np.mean(recuperada)
    numerador = float(np.sum(x_centrado * y_centrado))
    denominador = float(np.sqrt(np.sum(x_centrado ** 2) * np.sum(y_centrado ** 2)))
    return numerador / denominador


# Tabla
def imprimir_tabla(resultados: list[tuple[str, float, float]]) -> None:
    linea = "+-----------------+------------+------------------------+"
    print(linea)
    print(f"| {'Mensaje secreto':<15} | {'SNR (dB)':>10} | {'Correlación de Pearson':>22} |")
    print(linea)
    for nombre, snr_db, pearson in resultados:
        print(f"| {nombre:<15} | {snr_db:>10.4f} | {pearson:>22.15f} |")
    print(linea)


# Principal
def main(base_dir: str | Path | None = None) -> None:
    raiz = (
        Path(base_dir).resolve()
        if base_dir is not None
        else (Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd())
    )
    dir_originales = raiz / "audios" / "originales"
    dir_recuperados = raiz / "audios" / "recovered"

    resultados: list[tuple[str, float, float]] = []

    for i in range(1, 6):
        fs_orig, original = cargar_audio_mono(dir_originales / f"secret{i}.wav")
        fs_rec, recuperada = cargar_audio_mono(dir_recuperados / f"recovered_{i}.wav")

        if fs_orig != fs_rec or len(original) != len(recuperada):
            raise ValueError(
                f"Incompatibilidad en Secret {i}: original ({fs_orig} Hz, {len(original)}) "
                f"vs recuperado ({fs_rec} Hz, {len(recuperada)})."
            )

        snr_db = calcular_snr_db(original, recuperada)
        pearson = calcular_pearson(original, recuperada)
        resultados.append((f"Secret {i}", snr_db, pearson))

    imprimir_tabla(resultados)


# Arranque
if __name__ == "__main__":
    main()
