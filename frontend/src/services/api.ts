import axios from 'axios'
import type { Device, RiskRecord, Alert, HealthRecord } from '../types'

const api = axios.create({
  baseURL: '/api/v1',
  timeout: 10000,
})

// 设备API
export const deviceApi = {
  getAll: () => api.get<Device[]>('/devices'),
  getOne: (id: string) => api.get<Device>(`/devices/${id}`),
  getStatus: (id: string) => api.get(`/devices/${id}/status`),
  getCapabilities: (id: string) => api.get(`/devices/${id}/capabilities`),
}

// 视频流API
export const streamApi = {
  getPlayUrl: (deviceId: string) => api.get(`/streams/${deviceId}/play-url`),
  takeSnapshot: (deviceId: string) => api.post(`/streams/${deviceId}/snapshot`),
  start: (deviceId: string) => api.post(`/streams/${deviceId}/start`),
  stop: (deviceId: string) => api.post(`/streams/${deviceId}/stop`),
}

// 风险API
export const riskApi = {
  getCurrent: () => api.get<RiskRecord[]>('/risks/current'),
  getHistory: (params?: { device_id?: string; hours?: number }) =>
    api.get<RiskRecord[]>('/risks/history', { params }),
}

// 告警API
export const alertApi = {
  getAll: (params?: { status_filter?: string; device_id?: string; limit?: number }) =>
    api.get<Alert[]>('/alerts', { params }),
  confirm: (id: number, data: { handler: string; note?: string }) =>
    api.post<Alert>(`/alerts/${id}/confirm`, data),
  resolve: (id: number, data: { handler: string; note: string }) =>
    api.post<Alert>(`/alerts/${id}/resolve`, data),
  markFalseAlarm: (id: number, handler: string) =>
    api.patch<Alert>(`/alerts/${id}`, { status: 'FALSE_ALARM', handler }),
}

// 健康数据API
export const healthApi = {
  getLatest: (deviceId?: string) =>
    api.get<HealthRecord[]>('/health/latest', { params: { device_id: deviceId } }),
  getTrends: (deviceId: string, period: '24h' | '7d' | '30d') =>
    api.get(`/health/trends`, { params: { device_id: deviceId, period } }),
}

export default api
