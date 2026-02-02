import warnings
from astropy.io.fits.verify import VerifyWarning
warnings.simplefilter("ignore", category=VerifyWarning)

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits


FP = Path("data/exofop_spectra/TIC394137592/Spectrum_fits/TIC394137592S-ct20201119_1138.fits")


ORDER_IDX = None



def extract_wave_flux_from_primary(data: np.ndarray):
    """
    This CHIRON/ExoFOP-style file stores wavelength + spectrum in axis of length 2:
      header says: NAXIS1=2 / Axis 1 length: 0=wavelength, 1=spectrum

    Depending on how astropy arranges axes, data may appear as:
      - (n_orders, n_pix, 2)   [common numpy view]
      - (2, n_pix, n_orders)   [matches FITS dimension listing (2,3200,59)]
      - (2, n_orders, n_pix)   [less common]

    Returns:
      wave_img: (n_orders, n_pix)
      flux_img: (n_orders, n_pix)
    """
    if data.ndim != 3:
        raise RuntimeError(f"Expected 3D PrimaryHDU, got shape {data.shape}")

    # Case A: (n_orders, n_pix, 2)
    if data.shape[-1] == 2:
        wave_img = data[..., 0]
        flux_img = data[..., 1]
        return wave_img, flux_img

    # Case B: (2, n_pix, n_orders)  
    if data.shape[0] == 2:
        wave_img = data[0, :, :].T
        flux_img = data[1, :, :].T
        return wave_img, flux_img

    # Case C: (2, n_orders, n_pix) 
    if data.shape[0] == 2:
        wave_img = data[0, :, :]
        flux_img = data[1, :, :]
        return wave_img, flux_img

    raise RuntimeError(
        f"Could not interpret PrimaryHDU shape {data.shape}. "
        "Expected one axis of length 2 representing [wavelength, spectrum]."
    )


def main():
    if not FP.exists():
        raise FileNotFoundError(f"File not found: {FP.resolve()}")

    with fits.open(FP) as hdul:
        hdul.info()
        hdr = hdul[0].header
        data = np.asarray(hdul[0].data)

    print(f"\nPrimaryHDU raw numpy shape: {data.shape}")
    print(f"NAXIS1={hdr.get('NAXIS1')} (header comment indicates: 0=wavelength, 1=spectrum)")

    wave_img, flux_img = extract_wave_flux_from_primary(data)

    if wave_img.shape != flux_img.shape:
        raise RuntimeError(f"Wave/flux shapes differ: {wave_img.shape} vs {flux_img.shape}")

    n_orders, n_pix = flux_img.shape
    print(f"Interpreted wave_img/flux_img shape: {wave_img.shape} = (n_orders, n_pix)")

    # Choose order
    order_idx = (n_orders // 2) if ORDER_IDX is None else int(ORDER_IDX)
    if not (0 <= order_idx < n_orders):
        raise ValueError(f"ORDER_IDX must be between 0 and {n_orders-1}")

    wave = wave_img[order_idx, :].astype(float)
    flux = flux_img[order_idx, :].astype(float)

    # Clean
    good = np.isfinite(wave) & np.isfinite(flux)
    wave, flux = wave[good], flux[good]

    srt = np.argsort(wave)
    wave, flux = wave[srt], flux[srt]

    print(f"Order {order_idx}: wavelength range {wave.min():.2f} .. {wave.max():.2f} (N={len(wave)})")

    plt.figure()
    plt.plot(wave, flux, linewidth=1)
    plt.xlabel("Wavelength ")
    plt.ylabel("Flux ")
    plt.title(f"{FP.name} | order {order_idx}")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()


