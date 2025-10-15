import rasterio, numpy as np, pandas as pd
from rasterio.transform import from_bounds

path = "FWI.tif"
with rasterio.open(path) as src:
    band = src.read(1)
    h, w = band.shape

left, bottom, right, top = -25.0, 25.0, 45.0, 75.0
transform = from_bounds(left, bottom, right, top, w, h)

rows, cols = np.indices((h, w))
xs, ys = rasterio.transform.xy(transform, rows, cols)
lon = np.array(xs); lat = np.array(ys)

df = pd.DataFrame({"longitude": lon.ravel(),
                   "latitude":  lat.ravel(),
                   "FWI_value": band.ravel()}).dropna()
df.to_csv("FWI_extracted.csv", index=False)