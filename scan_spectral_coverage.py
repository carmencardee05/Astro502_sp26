import os
from glob import glob
import numpy as np
import pandas as pd
from astropy.io import fits



LINES = {
    "Li": 6707.8,
    "Halpha": 6562.8,
    "CaH": 3968.5,
    "CaK": 3933.7
}


def load_spectrum(file):

    try:
        hdul = fits.open(file)

        for hdu in hdul:

            if hasattr(hdu, "data") and hdu.data is not None:

                data = hdu.data

                # Case 1: table format
                if hasattr(data, "columns"):

                    colnames = [c.lower() for c in data.columns.names]

                    if "wavelength" in colnames and "flux" in colnames:
                        wl = data["wavelength"]
                        flux = data["flux"]
                        return wl, flux

                    if "wave" in colnames and "flux" in colnames:
                        wl = data["wave"]
                        flux = data["flux"]
                        return wl, flux

                # Case 2: image spectrum
                if isinstance(data, np.ndarray):

                    header = hdu.header

                    if "CRVAL1" in header and "CDELT1" in header:

                        start = header["CRVAL1"]
                        step = header["CDELT1"]

                        wl = start + step * np.arange(len(data))
                        flux = data

                        return wl, flux

        hdul.close()

    except Exception as e:
        print("Failed to read:", file)

    return None, None


# ============================================================
# Find spectra files
# ============================================================

print("Scanning organized spectra directory...")

all_files = glob("data/organized_spectra/**/*.fits*", recursive=True)

spectra_files = []

for f in all_files:

    if "_hdr" in f:
        continue

    if "_profile" in f:
        continue

    spectra_files.append(f)

print("Total candidate spectra:", len(spectra_files))




results = []

for spec_file in spectra_files:

    print("Processing:", spec_file)

    wl, flux = load_spectrum(spec_file)

    if wl is None:
        continue

    wl_min = np.nanmin(wl)
    wl_max = np.nanmax(wl)

    has_li = wl_min <= LINES["Li"] <= wl_max
    has_ha = wl_min <= LINES["Halpha"] <= wl_max
    has_cah = wl_min <= LINES["CaH"] <= wl_max
    has_cak = wl_min <= LINES["CaK"] <= wl_max

    results.append({
        "Filename": spec_file,
        "wavelength_low": wl_min,
        "wavelength_high": wl_max,
        "has_Li": has_li,
        "has_Halpha": has_ha,
        "has_CaH": has_cah,
        "has_CaK": has_cak
    })




df = pd.DataFrame(results)

df.to_csv("spectral_line_coverage.csv", index=False)

print("Saved results to spectral_line_coverage.csv")

print("Summary:")

print("Li coverage:", df["has_Li"].sum())
print("Halpha coverage:", df["has_Halpha"].sum())
print("CaH coverage:", df["has_CaH"].sum())
print("CaK coverage:", df["has_CaK"].sum())