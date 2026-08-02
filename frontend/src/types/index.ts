export interface Device {
  id: string
  device_serial: string
  device_type: 'camera' | 'radar'
  device_name?: string
  online_status: boolean
  capabilities: Record<string, boolean>
  location?: string
  last_seen_at?: string
  created_at: string
}

export interface RiskRecord {
  id: number
  device_id: string
  person_id?: string
  risk_type: string
  risk_score: number
  risk_level: 'LOW' | 'MEDIUM' | 'HIGH'
  status: string
  reasons: string[]
  snapshot_url?: string
  recorded_at: string
  created_at: string
}

export interface Alert {
  id: number
  device_id: string
  alert_type: string
  risk_level: 'LOW' | 'MEDIUM' | 'HIGH'
  message: string
  snapshot_url?: string
  status: 'NEW' | 'NOTIFIED' | 'CONFIRMED' | 'RESOLVED' | 'FALSE_ALARM' | 'EXPIRED'
  created_at: string
  confirmed_at?: string
  resolved_at?: string
  handler?: string
  handler_note?: string
}

export interface HealthRecord {
  id: number
  device_id: string
  heart_rate?: number
  respiratory_rate?: number
  in_bed?: boolean
  sleep_duration?: number
  leave_bed_count?: number
  sleep_quality_score?: number
  recorded_at: string
  created_at: string
}

export interface WebSocketMessage {
  type: 'risk_update' | 'alert' | 'health_update' | 'device_status'
  data: any
}
