from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import os


folder = "data/organized_spectra/TIC118327550"
region_min = 6695
region_max = 6725

# Helper
def extract_wave_flux(data):
    if data is None:
        return None, None, "No data"

    # 1D array
    if len(data.shape) == 1:
        flux = data
        wave = np.arange(len(flux))
        return wave, flux, "1D spectrum (pixel space)"

    # 2D array
    elif len(data.shape) == 2:
        flux = data[0]
        wave = np.arange(len(flux))
        return wave, flux, "2D echelle (first order, pixel space)"

    # 3D array
    elif len(data.shape) == 3:
        print("  -> Searching all orders for Lithium region")

        if data.shape[2] != 2:
            return None, None, f"3D unsupported last dimension: {data.shape[2]}"

        for i in range(data.shape[0]):
            wave_i = data[i, :, 0]
            flux_i = data[i, :, 1]

            mask = np.isfinite(wave_i) & np.isfinite(flux_i)
            wave_i = wave_i[mask]
            flux_i = flux_i[mask]

            if len(wave_i) == 0:
                continue

            if (np.min(wave_i) < 6708) and (np.max(wave_i) > 6708):
                print(f"  -> Found Lithium in order {i}")
                return wave_i, flux_i, f"3D extracted (order {i} contains Lithium)"

        return None, None, "3D extracted, but no order contains Lithium"

    else:
        return None, None, f"Unsupported shape: {data.shape}"

# Main

files = sorted(os.listdir(folder))

print("Files found:")
for f in files:
    print("  ", f)

plt.figure(figsize=(10, 6))

plotted = 0
ew_result = None

for file in files:
    path = os.path.join(folder, file)

    if not (file.endswith(".fits") or file.endswith(".fits.gz") or file.endswith(".fit")):
        continue

    print(f"\nTrying file: {file}")

    try:
        with fits.open(path) as hdul:
            data = hdul[0].data

            if data is None:
                print("  -> No data in primary HDU")
                continue

            print(f"  -> shape = {data.shape}")

            wave, flux, description = extract_wave_flux(data)
            print(f"  -> {description}")

            if wave is None or flux is None:
                continue

            mask = np.isfinite(wave) & np.isfinite(flux)
            wave = wave[mask]
            flux = flux[mask]

            if len(wave) == 0 or len(flux) == 0:
                continue

            # skip pixel-space spectra
            if np.max(wave) < 3000:
                print("  -> Skipping because this looks like pixel space")
                continue

            # zoom into Li region
            region_mask = (wave >= region_min) & (wave <= region_max)
            wave_region = wave[region_mask]
            flux_region = flux[region_mask]

            if len(wave_region) == 0:
                print("  -> No coverage in lithium region")
                continue

            # normalize by local median
            median_flux = np.median(flux_region)
            if not np.isfinite(median_flux) or median_flux == 0:
                print("  -> Bad median for normalization")
                continue

            flux_region_norm = flux_region / median_flux

            # smooth
            window = 25
            if len(flux_region_norm) > window:
                flux_region_smooth = np.convolve(
                    flux_region_norm,
                    np.ones(window) / window,
                    mode="same"
                )
            else:
                flux_region_smooth = flux_region_norm

            # flatten continuum using linear fit
            p = np.polyfit(wave_region, flux_region_smooth, 1)
            continuum = np.polyval(p, wave_region)
            flux_flat = flux_region_smooth / continuum

            # equivalent width
            ew_mask = (wave_region >= 6707.0) & (wave_region <= 6709.0)
            wave_ew = wave_region[ew_mask]
            flux_ew = flux_flat[ew_mask]

            if len(wave_ew) > 1:
                ew = np.trapz(1 - flux_ew, wave_ew)
                ew_result = ew
                print(f"Equivalent Width (Li 6708 Å): {ew:.5f} Å")
            else:
                print("Not enough points for EW calculation")

            # plot the full flattened region
            plt.plot(wave_region, flux_flat, label=file[:22])
            plotted += 1
            print("  -> Plotted successfully")

            # stop after first good spectrum
            break

    except Exception as e:
        print(f"  -> Error: {e}")

plt.axvline(6708.0, linestyle="--", color="black", linewidth=1, label="Li 6708 Å")
plt.axhline(1.0, linestyle="--", color="gray", linewidth=1)
plt.xlabel("Wavelength (Å)")
plt.ylabel("Flattened Normalized Flux")
plt.title("Single Smoothed Lithium Spectrum for TIC118327550")
plt.legend(fontsize=8)
plt.tight_layout()
plt.show()

if ew_result is not None:
    print(f"\nFinal EW result: {ew_result:.5f} Å")
else:
    print("\nNo EW result was measured.")

print(f"Total spectra plotted: {plotted}")