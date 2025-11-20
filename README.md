# Sentinel-2 Satellite Analysis (NDVI, NDWI, Vegetation Risk Map)

This project processes **Sentinel-2 L2A satellite imagery** to generate NDVI, NDWI, and a vegetation risk heatmap for environmental monitoring. The workflow is optimized to run efficiently on **Google Colab** with minimal RAM usage.

---

## 📍 Area of Interest (AOI)
Tested on a sample region located in **Delhi, India** using a defined bounding box.

---

## 🚀 Features
- Automatic download of recent, cloud-free Sentinel-2 images
- Extraction of essential spectral bands:
  - **B08 (NIR)**
  - **B04 (Red)**
  - **B03 (Green)**
- Computation of:
  - **NDVI (Normalized Difference Vegetation Index)**
  - **NDWI (Normalized Difference Water Index)**
- Generation of a **Vegetation Risk Map**
- Color-coded vegetation health visualization:
  - 🟢 Healthy vegetation
  - 🟡 Moderate vegetation
  - 🔴 Stressed / low vegetation

---

## 📊 Outputs
The code generates the following images:
- `ndvi.png`
- `ndwi.png`
- `risk_map.png`

These outputs help identify vegetation quality, water presence, and land condition.

---

## 🧠 Why This Project Matters
This workflow demonstrates real-world **remote sensing**, **GIS**, and **environmental analysis** techniques using open-source tools and real satellite data. It is ideal for:
- Academic projects
- Hackathons
- Environmental monitoring tasks
- Jury evaluations

---

## 🛠️ Requirements
- Python 3
- Google Colab (recommended)
- Libraries:
  - rasterio
  - numpy
  - matplotlib
  - sentinelsat


## 📥 How to Run
1. Clone the repository
2. Open the notebook or script in Google Colab
3. Enter Copernicus API credentials
4. Run all cells to generate NDVI, NDWI, and Risk Map

---

## 📄 License
This project is open-source and free to use.
