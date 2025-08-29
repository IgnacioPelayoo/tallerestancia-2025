#!/usr/bin/env python3
"""
Generador de datos sintéticos para un ejemplo de PCA.
Genera datos creando variables latentes de dimensión baja y proyectándolas
a un espacio de mayor dimensión con ruido añadido.
Guarda un CSV con las características y un gráfico PNG con la proyección PCA.
"""

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def generate_data(n_samples: int, n_features: int, n_components: int, noise: float, random_state: int):
    rng = np.random.default_rng(random_state)
    # Latent low-dim signals
    Z = rng.normal(size=(n_samples, n_components))

    # Random projection matrix (n_components x n_features), orthonormal rows
    A = rng.normal(size=(n_components, n_features))
    # Orthonormalize using QR on transpose to get orthonormal columns, then transpose back
    q, _ = np.linalg.qr(A.T)
    W = q.T[:n_components, :]

    X_clean = Z.dot(W)
    X_noisy = X_clean + rng.normal(scale=noise, size=X_clean.shape)
    return X_noisy, Z


def save_csv(X: np.ndarray, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cols = [f"feature_{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=cols)
    df.to_csv(out_path, index=False)


def plot_pca(X: np.ndarray, Z: np.ndarray, out_png: str, dim: int = 2):
    """Dibuja y guarda la proyección PCA en 2D o 3D según `dim`.

    out_png: ruta de salida (se sobrescribe si ya existe).
    dim: 2 o 3
    """
    if dim not in (2, 3):
        raise ValueError("dim must be 2 or 3")

    pca = PCA(n_components=dim)
    X_p = pca.fit_transform(X)
    explained = pca.explained_variance_ratio_

    os.makedirs(os.path.dirname(out_png), exist_ok=True)

    # Color by first latent dimension (if available)
    if Z.shape[1] >= 1:
        cmap_vals = Z[:, 0]
    else:
        cmap_vals = np.arange(X.shape[0])

    if dim == 2:
        plt.figure(figsize=(7, 5))
        sc = plt.scatter(X_p[:, 0], X_p[:, 1], c=cmap_vals, cmap="viridis", s=25, alpha=0.8)
        plt.colorbar(sc, label="latent dim 0 (color)")
        plt.xlabel("PC 1")
        plt.ylabel("PC 2")
        plt.title(f"PCA (explained var: {explained[0]:.2f}, {explained[1]:.2f})")
        plt.tight_layout()
        plt.savefig(out_png, dpi=150)
        plt.close()
    else:
        # 3D plot
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        sc = ax.scatter(X_p[:, 0], X_p[:, 1], X_p[:, 2], c=cmap_vals, cmap="viridis", s=30, alpha=0.8)
        fig.colorbar(sc, ax=ax, label="latent dim 0 (color)")
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        ax.set_zlabel("PC 3")
        # Build a concise title including explained variance
        expl_txt = ", ".join(f"{v:.2f}" for v in explained[:3])
        ax.set_title(f"PCA 3D (explained var: {expl_txt})")
        plt.tight_layout()
        plt.savefig(out_png, dpi=150)
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generador de datos sintéticos para PCA")
    parser.add_argument("--n-samples", type=int, default=500, help="Número de muestras")
    parser.add_argument("--n-features", type=int, default=10, help="Dimensión observada (features)")
    parser.add_argument("--n-components", type=int, default=5, help="Número de componentes latentes")
    parser.add_argument("--noise", type=float, default=0.1, help="Desviación estándar del ruido")
    parser.add_argument("--random-state", type=int, default=42, help="Semilla aleatoria")
    parser.add_argument("--output", type=str, default="pca_example/output", help="Directorio de salida")
    parser.add_argument("--pca-dim", type=int, default=2, choices=(2, 3), help="Dimensión de la visualización PCA: 2 o 3")

    args = parser.parse_args()

    X, Z = generate_data(args.n_samples, args.n_features, args.n_components, args.noise, args.random_state)

    csv_path = os.path.join(args.output, "pca_data.csv")
    png_name = "pca_plot.png" if args.pca_dim == 2 else "pca_plot_3d.png"
    png_path = os.path.join(args.output, png_name)

    save_csv(X, csv_path)
    plot_pca(X, Z, png_path, dim=args.pca_dim)

    # Información al usuario
    print(f"Guardado CSV en: {csv_path}")
    print(f"Guardado plot PCA en: {png_path}")
    print(f"Shape X: {X.shape}; latent shape: {Z.shape}")

if __name__ == "__main__":
    main()
