import warnings
from astropy.io.fits.verify import VerifyWarning
warnings.simplefilter("ignore", category=VerifyWarning)

from pathlib import Path
import numpy as np
from astropy.io import fits

base = Path("data/exofop")

def hdu_kind(hdu):
    # Table HDU?
    if hasattr(hdu, "columns") and hdu.columns is not None:
        return "TABLE"
    # Image HDU?
    if hdu.data is not None and isinstance(hdu.data, np.ndarray):
        return "IMAGE"
    return "NONE"

for fp in sorted(base.rglob("*.fits")):
    print("\n" + str(fp))
    print("-" * 77)

    try:
        with fits.open(fp) as hdul:
            for i, hdu in enumerate(hdul):
                kind = hdu_kind(hdu)

                if kind == "IMAGE":
                    print(f"HDU {i}: {type(hdu).__name__} → IMAGE shape={hdu.data.shape}")
                elif kind == "TABLE":
                    cols = [c.name for c in hdu.columns]
                    print(f"HDU {i}: {type(hdu).__name__} → TABLE ncols={len(cols)} cols={cols[:10]}")
                else:
                    print(f"HDU {i}: {type(hdu).__name__} → NONE")

    except Exception as e:
        print(f"[ERROR] Could not open: {e}")
