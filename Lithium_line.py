from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import glob


folder = "data/koa/HIRES/KELT-1/lev1/extracted/HI.20170702.51878"
region_min = 6650
region_max = 6750
target_wave = 6708.0

v_rad = 65.0
c = 299792.458
z = v_rad / c

def get_observed_wavelength(v_rad, target_wave):
    
    c = 299792.458
    z = v_rad / c
    lambda_obs = target_wave * (1 + z)
    
    return lambda_obs
    
    wavelength_rest = lambda_obs / (1 + z)
    
    return wavelength_rest



def get_wave_flux_from_table(table, colnames):
    upper_names = [name.upper() for name in colnames]

    # Common possibilities
    wave_candidates = ["WAVELENGTH", "WAVE", "LAMBDA", "WL"]
    flux_candidates = ["FLUX", "SPEC", "SPECTRUM", "SCI", "COUNTS"]

    wave = None
    flux = None

    for cand in wave_candidates:
        if cand in upper_names:
            wave = table[colnames[upper_names.index(cand)]]
            break

    for cand in flux_candidates:
        if cand in upper_names:
            flux = table[colnames[upper_names.index(cand)]]
            break

    # Fallback: use first two columns
    if wave is None or flux is None:
        print("  -> Could not identify standard column names, trying first two columns")
        wave = table.field(0)
        flux = table.field(1)

    return np.array(wave), np.array(flux)


files = sorted(glob.glob(f"{folder}/*.fits.gz"))

print(f"Found {len(files)} files")

found_any = False


for file in files:
    print(f"\nTrying: {file}")

    try:
        with fits.open(file) as hdul:
            hdul.info()

            # Use extension 1 since this is a BinTable
            if len(hdul) < 2 or hdul[1].data is None:
                print("  -> No table data in HDU 1")
                continue

            table = hdul[1].data
            colnames = table.columns.names

            print("  -> Column names:")
            for name in colnames:
                print("     ", name)

            wave, flux = get_wave_flux_from_table(table, colnames)
            
          

            # Flatten in case columns are weird shapes
            wave = np.ravel(wave)
            flux = np.ravel(flux)
            
            # Apply doppler correction
            wave = wave / (1 + z)

            # Clean
            mask = np.isfinite(wave) & np.isfinite(flux)
            wave = wave[mask]
            flux = flux[mask]

            if len(wave) == 0 or len(flux) == 0:
                print("  -> No finite wave/flux values")
                continue

            print(f"  -> Wavelength range: {np.min(wave):.2f} to {np.max(wave):.2f} Å")
            
            wave_rest = wave / (1 + z)
            

            # Check if lithium region is present
            if not ((np.min(wave) < target_wave) and (np.max(wave) > target_wave)):
                print("  -> This file does not cover 6708 Å")
                continue

            # Zoom into lithium region
            region_mask = (wave >= region_min) & (wave <= region_max)
            wave_region = wave[region_mask]
            flux_region = flux[region_mask]

            if len(wave_region) == 0:
                print("  -> No data in lithium window")
                continue

            # Sort just in case wavelength is unordered
            sort_idx = np.argsort(wave_region)
            wave_region = wave_region[sort_idx]
            flux_region = flux_region[sort_idx]

            # Normalize
            median_flux = np.median(flux_region)
            if not np.isfinite(median_flux) or median_flux == 0:
                print("  -> Bad median flux")
                continue

            flux_norm = flux_region / median_flux

            # Smooth
            window = 15
            if len(flux_norm) > window:
                flux_smooth = np.convolve(
                    flux_norm,
                    np.ones(window) / window,
                    mode="same"
                )
            else:
                flux_smooth = flux_norm

            # Plot
            plt.figure(figsize=(8, 5))
            plt.plot(wave_region, flux_smooth, linewidth=1.2)
            plt.axvline(target_wave, color="red", linestyle="--", label="Li 6708 Å")
            plt.axhline(1.0, color="gray", linestyle="--")

            plt.xlim(region_min, region_max)
            plt.xlabel("Wavelength (Å)")
            plt.ylabel("Normalized Flux")
            plt.title(f"HIRES Lithium Region\n{file.split('/')[-1]}")
            plt.legend()
            plt.tight_layout()
            plt.show()

            print("  -> Plotted successfully")
            found_any = True
            break

    except Exception as e:
        print(f"  -> Error: {e}")

if not found_any:
    print("\nNo file with usable lithium-region data was plotted.")