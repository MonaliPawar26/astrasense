# Low-RAM Sentinel-2 NDVI / NDWI pipeline — FULL UPDATED (Colab-ready)
# - Uses Earth Search STAC (Element84) for Sentinel-2 L2A COGs
# - Reprojects AOI (EPSG:4326) -> COG CRS before windowing (fixes "Intersection empty")
# - Reads only the window overlapping AOI (server-side windowed reads)
# - Keeps memory low (MAX_PIXELS cap)
# - Produces NDVI, NDWI, risk map, overlay, .zip for download

# 0) Install required packages (run once)
print("Installing packages — please wait...")
!pip install --quiet pystac-client rasterio matplotlib numpy requests tqdm shapely pyproj opencv-python-headless

# -----------------------
# 1) Imports & config
# -----------------------
import os, uuid, shutil
from datetime import datetime, timedelta
from urllib.parse import urlparse
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import from_bounds
from pystac_client import Client

from shapely.geometry import box
from shapely.ops import transform
import pyproj

import cv2

# Output dirs
OUT_DIR = "outputs"
DL_DIR = "downloads"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(DL_DIR, exist_ok=True)

# STAC endpoint
EARTH_SEARCH_STAC = "https://earth-search.aws.element84.com/v1/"

# Low-memory safety: limit of pixels read (rows * cols). Tune downward for less RAM.
MAX_PIXELS = 1024 * 1024  # ~1 million pixels (e.g., 1000x1000)

# -----------------------
# 2) Helpers: STAC search & asset utils
# -----------------------
def open_stac_client():
    return Client.open(EARTH_SEARCH_STAC)

def resolve_date_token(tok):
    tok = str(tok)
    if tok.upper() == "NOW":
        return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    if tok.upper().startswith("NOW-") and "DAYS" in tok.upper():
        try:
            n = int(tok.upper().split("NOW-")[1].split("DAYS")[0])
            return (datetime.utcnow() - timedelta(days=n)).strftime("%Y-%m-%dT%H:%M:%SZ")
        except Exception:
            return tok
    return tok

def find_best_item_for_aoi(aoi_bbox, start, end, cloud_max=10, limit=50):
    """
    Returns a pystac Item for the AOI and date-range (best by cloudcover <= cloud_max or best available).
    aoi_bbox: [minLon, minLat, maxLon, maxLat]
    """
    client = open_stac_client()
    search = client.search(collections=["sentinel-2-l2a"], bbox=tuple(aoi_bbox), datetime=f"{start}/{end}", limit=limit)
    items = list(search.get_items())
    if not items:
        raise RuntimeError("No sentinel-2-l2a items found for the AOI/date window. Try expanding the date range or increasing cloud_max.")
    rows = []
    for it in items:
        cc = it.properties.get("eo:cloud_cover")
        cc = float(cc) if cc is not None else 999.0
        rows.append((cc, it))
    rows.sort(key=lambda x: x[0])
    for cc, it in rows:
        if cc <= cloud_max:
            return it
    return rows[0][1]

def get_asset_href_safe(asset):
    """Return href for pystac.Asset, tolerant of shapes."""
    href = getattr(asset, "href", None)
    if href:
        return href
    # fallback to extra_fields links if present
    extra = getattr(asset, "extra_fields", None) or {}
    # try common keys
    for k in ("href", "url", "uri"):
        v = extra.get(k)
        if v:
            return v
    return None

def find_band_href(item, band_tokens):
    """Search item.assets for a band asset matching band_tokens (tokens list)."""
    for key, asset in item.assets.items():
        href = get_asset_href_safe(asset)
        # 1) check asset key
        if key and any(tok.upper() in key.upper() for tok in band_tokens):
            if href:
                return href
        # 2) check filename/href
        if href:
            fname = os.path.basename(href).upper()
            if any(tok.upper() in fname for tok in band_tokens):
                return href
        # 3) check eo:bands metadata in extra_fields
        extra = getattr(asset, "extra_fields", None) or {}
        eo_bands = extra.get("eo:bands") or []
        for b in eo_bands:
            name = (b.get("name") or b.get("common_name") or "").upper()
            if any(tok.upper() in name for tok in band_tokens):
                if href: return href
    return None

def get_b03_b04_b08_hrefs(item):
    """Return dictionary of HTTP(S) hrefs for B03,B04,B08 or raise helpful error."""
    bands = {
        "B03": ["B03", "GREEN", "BAND_3"],
        "B04": ["B04", "RED", "BAND_4"],
        "B08": ["B08", "NIR", "BAND_8"]
    }
    hrefs = {}
    missing = []
    for b, toks in bands.items():
        href = find_band_href(item, toks)
        if not href:
            missing.append(b)
        else:
            scheme = urlparse(href).scheme
            if scheme not in ("http", "https"):
                raise RuntimeError(
                    f"Band {b} asset is not HTTP/HTTPS (href={href}). This script expects HTTP(S) COG/TIFF assets. "
                    "If the item only exposes s3:// or gs:// assets, enable the S3 fallback or download the SAFE manually."
                )
            hrefs[b] = href
    if missing:
        raise RuntimeError(f"Could not find COG/TIFF assets for bands: {missing}. Available asset keys: {list(item.assets.keys())}")
    return hrefs

# -----------------------
# 3) Low-memory reader: reproject AOI -> COG CRS, windowed read with downsample if needed
# -----------------------
def read_aoi_window_from_cog(href, aoi_bbox, max_pixels=MAX_PIXELS):
    """
    Read only the pixel window overlapping `aoi_bbox` (WGS84 lon/lat) from COG `href`.
    Returns: (arr (float32 2D), meta dict)
    Meta includes: window, shape, src_crs
    """
    with rasterio.Env():
        with rasterio.open(href) as src:
            # 1) Create AOI polygon in WGS84
            aoi_poly_wgs84 = box(*aoi_bbox)  # lon/lat

            # 2) Reproject AOI polygon to dataset CRS
            try:
                transformer = pyproj.Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
            except Exception as e:
                raise RuntimeError(f"Failed to create transformer to dataset CRS ({src.crs}): {e}")

            project = transformer.transform
            aoi_poly_dst = transform(project, aoi_poly_wgs84)
            minx, miny, maxx, maxy = aoi_poly_dst.bounds

            # 3) Create rasterio window from bounds in dataset CRS
            win = from_bounds(minx, miny, maxx, maxy, transform=src.transform)
            win = win.round_offsets().round_lengths()
            # clamp to image bounds
            win = win.intersection(rasterio.windows.Window(0, 0, src.width, src.height))

            if win.width <= 0 or win.height <= 0:
                raise RuntimeError(
                    "AOI does not intersect this COG. After reprojection to dataset CRS the window is empty. "
                    f"AOI bounds in dataset CRS: {(minx, miny, maxx, maxy)}"
                )

            rows = int(win.height); cols = int(win.width)

            # 4) If window too large, read downsampled out_shape to cap memory
            if rows * cols > max_pixels:
                scale = (rows * cols) / float(max_pixels)
                factor = np.sqrt(scale)
                out_rows = max(1, int(rows / factor))
                out_cols = max(1, int(cols / factor))
                arr = src.read(1, window=win, out_shape=(out_rows, out_cols), resampling=Resampling.bilinear).astype(np.float32)
            else:
                arr = src.read(1, window=win).astype(np.float32)

            meta = {"window": win, "shape": arr.shape, "src_crs": str(src.crs)}
            return arr, meta

# -----------------------
# 4) Index computations & visualization (small arrays)
# -----------------------
def auto_scale(arr):
    # Sentinel-2 L2A reflectances often scaled by 10000
    arr = np.asarray(arr, dtype=np.float32)
    valid = arr[~np.isnan(arr)]
    if valid.size == 0:
        return arr
    if np.nanpercentile(valid, 98) > 1000:
        return arr / 10000.0
    return arr

def compute_ndvi(nir, red, eps=1e-6):
    return (nir - red) / (nir + red + eps)

def compute_ndwi(green, nir, eps=1e-6):
    return (green - nir) / (green + nir + eps)

def save_png_heatmap(arr, path, cmap="RdYlGn", vmin=None, vmax=None):
    plt.figure(figsize=(6,6), dpi=150)
    valid = arr[~np.isnan(arr)]
    if valid.size == 0:
        plt.close()
        return
    if vmin is None: vmin = np.nanpercentile(valid, 2)
    if vmax is None: vmax = np.nanpercentile(valid, 98)
    plt.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.axis('off'); plt.tight_layout()
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close()

def overlay_small_rgb(b03, b04, b08, index_arr, outpath, alpha=0.45):
    def norm(x):
        v = x[~np.isnan(x)]
        if v.size == 0: return np.zeros_like(x, dtype=np.uint8)
        mn, mx = np.nanpercentile(v, 2), np.nanpercentile(v, 98)
        y = np.clip((x - mn) / (mx - mn + 1e-9), 0, 1)
        return (y * 255).astype(np.uint8)
    r = norm(b04); g = norm(b03); b = norm(b08)
    rgb = np.dstack([r,g,b])
    idx_norm = (index_arr - np.nanmin(index_arr)) / (np.nanmax(index_arr) - np.nanmin(index_arr) + 1e-9)
    cmap = plt.get_cmap("jet")
    heat = (cmap(idx_norm)[:,:,:3] * 255).astype(np.uint8)
    if heat.shape[:2] != rgb.shape[:2]:
        heat = cv2.resize(heat, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
    overlay = cv2.addWeighted(heat, alpha, rgb[..., ::-1], 1-alpha, 0)
    cv2.imwrite(outpath, overlay)
    return outpath

# -----------------------
# 5) Main runner: low-ram AOI flow
# -----------------------
def run_lowram_analysis(params):
    """
    params keys:
      - aoi: [minLon, minLat, maxLon, maxLat] (required)
      - start: 'YYYY-MM-DD' or 'NOW-60DAYS'
      - end: 'YYYY-MM-DD' or 'NOW'
      - cloud_max: int
    Returns: dict with summary, assets, zip or {'error': msg}
    """
    try:
        aoi = params.get("aoi")
        if not aoi or len(aoi) != 4:
            raise ValueError("Provide a valid aoi [minLon, minLat, maxLon, maxLat].")
        start = resolve_date_token(params.get("start", "NOW-60DAYS"))
        end = resolve_date_token(params.get("end", "NOW"))
        cloud_max = int(params.get("cloud_max", 10))

        # 1) choose best STAC item
        item = find_best_item_for_aoi(aoi, start, end, cloud_max=cloud_max)

        # 2) get COG hrefs for B03,B04,B08
        hrefs = get_b03_b04_b08_hrefs(item)

        # 3) read small windows for each band (reproject AOI -> dataset CRS inside function)
        b03_arr, meta = read_aoi_window_from_cog(hrefs["B03"], aoi)
        b04_arr, _ = read_aoi_window_from_cog(hrefs["B04"], aoi)
        b08_arr, _ = read_aoi_window_from_cog(hrefs["B08"], aoi)

        # 4) autoscale to reflectance
        b03 = auto_scale(b03_arr); b04 = auto_scale(b04_arr); b08 = auto_scale(b08_arr)

        # 5) indices
        ndvi = compute_ndvi(b08, b04)
        ndwi = compute_ndwi(b03, b08)
        ndvi_norm = (ndvi + 1) / 2.0
        ndwi_norm = (ndwi + 1) / 2.0
        risk_map = np.clip(0.7 * ndwi_norm + 0.6 * (1 - ndvi_norm), 0, 1)

        # 6) save outputs
        uid = uuid.uuid4().hex[:8]
        ndvi_path = os.path.join(OUT_DIR, f"ndvi_{uid}.png")
        ndwi_path = os.path.join(OUT_DIR, f"ndwi_{uid}.png")
        risk_path = os.path.join(OUT_DIR, f"risk_{uid}.png")
        overlay_path = os.path.join(OUT_DIR, f"overlay_{uid}.png")

        save_png_heatmap(ndvi, ndvi_path, cmap="RdYlGn", vmin=-1, vmax=1)
        save_png_heatmap(ndwi, ndwi_path, cmap="Blues", vmin=-1, vmax=1)
        save_png_heatmap(risk_map, risk_path, cmap="jet", vmin=0, vmax=1)
        overlay_small_rgb(b03, b04, b08, risk_map, overlay_path, alpha=0.45)

        # 7) classification summary
        class_map = np.zeros_like(ndvi, dtype=np.uint8)
        class_map[(ndwi_norm > 0.55) & (risk_map > 0.5)] = 1
        class_map[(ndvi < 0.15) & (risk_map > 0.4)] = 2
        class_map[(ndvi < -0.1) & (risk_map > 0.4)] = 3

        total = class_map.size
        counts = {
            "normal_pct": float(np.sum(class_map==0)/total),
            "flood_pct": float(np.sum(class_map==1)/total),
            "drought_pct": float(np.sum(class_map==2)/total),
            "vegloss_pct": float(np.sum(class_map==3)/total)
        }

        top_threat = max([
            ("Flood", counts["flood_pct"]),
            ("Drought", counts["drought_pct"]),
            ("Vegetation Loss", counts["vegloss_pct"]),
            ("Normal", counts["normal_pct"])
        ], key=lambda x: x[1])[0]

        # 8) zip outputs
        zip_base = f"/content/analysis_assets_{uuid.uuid4().hex[:8]}"
        zip_path = shutil.make_archive(zip_base, 'zip', root_dir='.', base_dir=OUT_DIR)

        return {
            "summary": {
                "top_threat": top_threat,
                "mean_ndvi": float(np.nanmean(ndvi)),
                "mean_ndwi": float(np.nanmean(ndwi)),
                "mean_risk": float(np.nanmean(risk_map)),
                **counts
            },
            "assets": {
                "ndvi_image": ndvi_path,
                "ndwi_image": ndwi_path,
                "risk_image": risk_path,
                "overlay": overlay_path
            },
            "zip": zip_path
        }

    except Exception as e:
        return {"error": str(e)}

# -----------------------
# 6) Colab display helper
# -----------------------
def display_results_colab(result):
    from IPython.display import display, Image, Markdown, FileLink
    if "error" in result:
        display(Markdown("## 🛑 Error"))
        display(Markdown(f"`{result['error']}`"))
        return
    s = result["summary"]
    display(Markdown("## ✨ Analysis Summary"))
    display(Markdown(f"**Top threat:** {s['top_threat']}"))
    display(Markdown(f"- Mean NDVI: {s['mean_ndvi']:.3f}"))
    display(Markdown(f"- Mean NDWI: {s['mean_ndwi']:.3f}"))
    display(Markdown(f"- Mean risk: {s['mean_risk']:.3f}"))
    display(Markdown("### Images"))
    for k, v in result["assets"].items():
        if os.path.exists(v):
            display(Markdown(f"#### {os.path.basename(v)}"))
            display(Image(v))
    if os.path.exists(result.get("zip","")):
        display(Markdown("### Download"))
        display(FileLink(result["zip"]))

# -----------------------
# Example usage (update the AOI to your small bbox)
# -----------------------
if __name__ == "__main__":
    params = {
        # Replace with your small AOI (lon/lat)
        "aoi": [32.783, 34.775, 32.79, 34.78],
        "start": "NOW-30DAYS",
        "end": "NOW",
        "cloud_max": 15
    }
    print("Running low-RAM analysis (small AOI)...")
    out = run_lowram_analysis(params)
    try:
        display_results_colab(out)
    except Exception:
        print(out)
