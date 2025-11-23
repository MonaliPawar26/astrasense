import { TeamMember, Region, RiskLevel } from './types';

export const APP_NAME = "ASTRASENSE";
export const TEAM_NAME = "ShadowHack";

export const TEAM_MEMBERS: TeamMember[] = [
  { name: "Monali Pawar", branch: "CSE", email: "monapawar0926@gmail.com", contact: "9175879234" },
  { name: "Pranjal Navgale", branch: "CSE", email: "pranjalnavgale@gmail.com", contact: "8625019446" },
  { name: "Pranav Patil", branch: "CSE", email: "patilpranav1012@gmail.com", contact: "7420871303" },
  { name: "Nikhil Rathod", branch: "CSE", email: "rathodnikhil2006@gmail.com", contact: "9175879845" },
  { name: "Janhavi Pawar", branch: "DS", email: "janhavipawar345@gmail.com", contact: "9325836893" }
];

export const MOCK_REGIONS: Region[] = [
  {
    id: "mah-001",
    name: "Vidarbha Agricultural Zone",
    coordinates: "21.1458° N, 79.0882° E",
    imageUrl: "https://picsum.photos/seed/sat1/800/600",
    currentRisk: RiskLevel.HIGH
  },
  {
    id: "ker-002",
    name: "Kuttanad Wetland Region",
    coordinates: "9.4230° N, 76.4710° E",
    imageUrl: "https://picsum.photos/seed/sat2/800/600",
    currentRisk: RiskLevel.MODERATE
  },
  {
    id: "raj-003",
    name: "Thar Desert Border",
    coordinates: "26.9124° N, 70.9373° E",
    imageUrl: "https://picsum.photos/seed/sat3/800/600",
    currentRisk: RiskLevel.CRITICAL
  }
];

export const SATELLITE_SOURCES = [
  "MODIS (Moderate Resolution Imaging Spectroradiometer)",
  "Sentinel-2 (High-resolution optical imagery)",
  "Landsat-8 (Thermal & Operational Land Imager)",
  "ISRO Bhuvan Portal Integration"
];

// Mock historical data for charts
export const MOCK_SENSOR_HISTORY = [
  { time: '06:00', ndvi: 0.65, ndwi: 0.2, lst: 28 },
  { time: '08:00', ndvi: 0.62, ndwi: 0.18, lst: 30 },
  { time: '10:00', ndvi: 0.58, ndwi: 0.15, lst: 35 },
  { time: '12:00', ndvi: 0.55, ndwi: 0.12, lst: 42 },
  { time: '14:00', ndvi: 0.50, ndwi: 0.10, lst: 45 },
  { time: '16:00', ndvi: 0.52, ndwi: 0.11, lst: 40 },
  { time: '18:00', ndvi: 0.58, ndwi: 0.14, lst: 34 },
];