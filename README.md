

---

# 🛰️ ASTRASENSE

## 🚨 AI-Powered Early Hazard & Environmental Risk Detection

**Leveraging Multi-Modal Geospatial Data for Climate Resilience**

---

### Theme: **AI/ML for Space Data Interpretation**  
Transforming Remote Sensing into Actionable Environmental Intelligence

⚡ Real-Time Inference · Hybrid AI Modeling · Proactive Disaster Mitigation

---

## 🚀 Overview

**ASTRASENSE** is an advanced AI-driven platform that converts vast streams of **multi-satellite imagery** into accurate, real-time environmental risk indices and early hazard warnings. Automating the entire geospatial analytics pipeline, it provides critical decision support for governments, disaster response agencies, and agricultural sectors.

**Key Objectives:**
- ⚠️ **Proactive Hazard Detection:** Early identification of floods, droughts, and environmental anomalies.
- 🌾 **Agricultural Resilience:** Micro-localized alerts to minimize crop losses.
- 🗺️ **Strategic Planning:** Supporting disaster response and climate resilience infrastructure.
- 📈 **Operational Efficiency:** Reducing manual satellite data interpretation efforts.

---

## 🎯 Core Features

### 🧠 Hybrid AI Engine
- **Innovative Architecture:** Combines **CNN, LSTM, and Autoencoder** models for superior spatio-temporal analysis and anomaly detection.

### 🔗 Multi-Satellite Data Fusion
- **Robust Integration:** Seamlessly fuses data from **MODIS, Sentinel, Landsat** to address cloud cover and data gaps.

### 🗺️ Region-Specific Risk Modeling
- **Localized Calibration:** Models adapt dynamically to diverse terrains and micro-climates, ensuring highly relevant risk predictions.

### ⏱️ Automated End-to-End Pipeline
- **Scalable Workflow:** Automates data ingestion, preprocessing, index computation (NDVI, NDWI, NDSI), and alert dissemination.

### 🚨 Smart Risk Scoring System
- **Predictive Analytics:** Quantifies hazard likelihood and severity for early warning rather than post-event detection.

---

## 🧩 Problem Statement

### Addressing Climate Vulnerability through Space-Based Assets

As climate events intensify, the need for reliable, rapid intelligence becomes urgent:
- 🌊 **Floods:** Increasingly frequent and severe, causing economic and human losses.
- 🏜️ **Droughts:** Threatening food security and rural livelihoods.
- ☁️ **Data Latency & Cloud Cover:** Traditional monitoring is hampered by cloud contamination and manual analysis delays.

**ASTRASENSE** bridges this gap, providing fast, explainable, scalable insights, transforming raw satellite data into life-saving decisions.

---

## 🧭 System Architecture & Workflow

The system operates via an automated, high-throughput pipeline:

1. **Data Ingestion:** Continuous acquisition from satellite sources.
2. **Preprocessing & Indexing:** Cloud masking, geo-alignment, normalization, calculation of indices (NDWI, NDSI).
3. **AI Analysis:** Hybrid AI models analyze multi-temporal, multi-spectral data for anomalies.
4. **Risk Modeling:** Geospatial analytics generate a **Smart Risk Score** indicating hazard probability and severity.
5. **Alerting & Distribution:** Real-time alerts via Dashboard, API, and SMS channels.

---

## 🧬 Tech Stack

### 🧠 AI & Modeling
- **Architecture:** Hybrid CNN-LSTM-Autoencoder
- **Frameworks:** TensorFlow, Keras

### 🛰️ Data & Geospatial Processing
- **Sources:** MODIS, Sentinel, Landsat (Open & Free)
- **Tools:** GDAL, Rasterio, GeoPandas

### 💻 Backend
- **Language:** Python
- **Framework:** FastAPI / Flask
- **Databases:** PostgreSQL with PostGIS, MongoDB (for spatial data)
- **Containerization:** Docker, Kubernetes

### 🌐 Frontend
- **Technologies:** HTML, CSS, JavaScript
- **Visualization:** Mapbox GL JS, Leaflet, D3.js
- **Features:** Interactive maps, risk dashboards, real-time alerts

### ☁️ Cloud & Deployment
- **Cloud Providers:** AWS, Google Cloud, Azure
- **Services:** AWS Lambda, Google Cloud Functions (Serverless)
- **CI/CD:** Jenkins, GitHub Actions
- **Monitoring:** Prometheus, Grafana

### 📡 Additional Tools & Integrations
- **Data Pipelines:** Apache Airflow, Kafka
- **Notification Gateway:** Twilio API (SMS), Push Notification Services
- **Version Control:** Git, GitHub

---
# Dashboard Screenshots

## Main Dashboard View
<img width="1359" height="607" alt="Image" src="https://github.com/user-attachments/assets/a1d7b69e-d38a-4684-88c8-4496428bf602" />
## AI Analysis Panel


### 🧪 Sample Risk Alert

```json
{
  "hazard_type": "Flood Risk",
  "location_sector": "Ganga River Basin, Sub-region 4",
  "risk_score": 8.5,
  "prediction_confidence": 0.95,
  "index_anomaly": "NDWI: +0.45 (Indicates high water content)",
  "recommended_action": "High probability. Activate local contingency plans and issue evacuation advisories for low-lying areas."
}
```

---

## Development Workflow

- **Data Pipeline:** Automated multi-satellite ingestion and fusion.
- **Model Engineering:** Designing, training, and fine-tuning the Hybrid AI architecture.
- **Risk Scoring:** Developing and validating the Smart Risk Score.
- **Deployment:** Building APIs, dashboards, and notification gateways for real-time access.

---

## Use Cases

- **Proactive Disaster Management:** Early warnings up to 72 hours ahead.
- **Precision Agriculture:** Localized drought and flood indices for resource optimization.
- **Infrastructure Safety:** Environmental risk assessments for infrastructure projects.
- **Climate Resilience:** Data-driven policy formulation and regional planning.

---

## 🔮 Future Enhancements

- Integration of ground sensors and IoT data for enhanced validation.
- Advanced Spatio-Temporal Graph Neural Networks for complex hazard modeling.
- Dynamic cloud scaling and serverless deployment for global reach.
- User-configurable risk thresholds and customizable alert zones.

---

## 👤 Development Team

**Team ShadowHack**  
📧 monapawar0926@gmail.com 


**Members:**
- Monali Pawar
- Pranjal Navgale
- Pranav Patil
- Nikhil Rathod
- Janhavi Pawar

---

## 💚 ASTRASENSE — Transforming Satellite Data Into Life-Saving Environmental Intelligence

---
