from astropy.io import fits
with fits.open('/Users/carmencardenas/Desktop/TIC262530407S-ct20190818_1170.fits') as hdul:
    hdul.info()  # Displays the file structure and info
    data = hdul[0].data # Access data from the primary HDU
    header = hdul[0].header # Access the header

print("\nHeader Information:")
print(repr(header)) # Prints header keywords and values in a readable format

print("\nData (first few rows/pixels):")
print(data) # Prints the actual data array (be cautious with large files)

