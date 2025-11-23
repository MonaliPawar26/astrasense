export enum RiskLevel {
  LOW = 'Low',
  MODERATE = 'Moderate',
  HIGH = 'High',
  CRITICAL = 'Critical'
}

export interface TeamMember {
  name: string;
  branch: string;
  email: string;
  contact: string;
}

export interface SensorData {
  time: string;
  ndvi: number; // Normalized Difference Vegetation Index
  ndwi: number; // Normalized Difference Water Index
  lst: number;  // Land Surface Temperature
}

export interface Alert {
  id: string;
  type: 'Flood' | 'Drought' | 'Fire' | 'Degradation';
  severity: RiskLevel;
  location: string;
  timestamp: string;
  message: string;
}

export interface Region {
  id: string;
  name: string;
  coordinates: string;
  imageUrl: string;
  currentRisk: RiskLevel;
}