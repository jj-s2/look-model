import { useState, useEffect } from 'react'
import { Card, Select, Tag, Progress, Space, Empty } from 'antd'
import { VideoCameraOutlined, WarningOutlined } from '@ant-design/icons'
import { deviceApi, streamApi, riskApi } from '../services/api'
import { useWebSocket } from '../hooks/useWebSocket'
import type { Device, RiskRecord } from '../types'

export default function LiveMonitor() {
  const [devices, setDevices] = useState<Device[]>([])
  const [selectedDevice, setSelectedDevice] = useState<string>('')
  const [playUrl, setPlayUrl] = useState<string>('')
  const [currentRisk, setCurrentRisk] = useState<RiskRecord | null>(null)
  const { lastMessage } = useWebSocket('live')

  useEffect(() => {
    loadDevices()
  }, [])

  useEffect(() => {
    if (selectedDevice) {
      loadPlayUrl()
      loadCurrentRisk()
    }
  }, [selectedDevice])

  useEffect(() => {
    if (lastMessage?.type === 'risk_update' && lastMessage.data.device_id === selectedDevice) {
      setCurrentRisk(lastMessage.data as RiskRecord)
    }
  }, [lastMessage, selectedDevice])

  const loadDevices = async () => {
    try {
      const res = await deviceApi.getAll()
      const cameras = res.data.filter(d => d.device_type === 'camera')
      setDevices(cameras)
      if (cameras.length > 0) {
        setSelectedDevice(cameras[0].id)
      }
    } catch (error) {
      console.error('加载设备失败:', error)
    }
  }

  const loadPlayUrl = async () => {
    try {
      const res = await streamApi.getPlayUrl(selectedDevice)
      setPlayUrl(res.data.play_url)
    } catch (error) {
      console.error('获取播放地址失败:', error)
    }
  }

  const loadCurrentRisk = async () => {
    try {
      const res = await riskApi.getHistory({ device_id: selectedDevice, hours: 1 })
      if (res.data.length > 0) {
        setCurrentRisk(res.data[0])
      }
    } catch (error) {
      console.error('获取风险数据失败:', error)
    }
  }

  const getRiskColor = (level: string) => {
    if (level === 'HIGH') return 'red'
    if (level === 'MEDIUM') return 'orange'
    return 'green'
  }

  return (
    <div>
      <h1>实时监控</h1>

      <Card style={{ marginBottom: 24 }}>
        <Space>
          <span>选择设备：</span>
          <Select
            style={{ width: 300 }}
            value={selectedDevice}
            onChange={setSelectedDevice}
            options={devices.map(d => ({
              label: d.device_name || d.device_serial,
              value: d.id,
            }))}
          />
        </Space>
      </Card>

      <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: 24 }}>
        <Card title="实时视频" extra={<VideoCameraOutlined />}>
          {playUrl ? (
            <div style={{ 
              background: '#000', 
              aspectRatio: '16/9', 
              display: 'flex', 
              alignItems: 'center', 
              justifyContent: 'center',
              color: '#fff',
              fontSize: 18
            }}>
              <div>
                <p>视频播放器</p>
                <p style={{ fontSize: 14, color: '#999' }}>{playUrl}</p>
                <p style={{ fontSize: 12, marginTop: 16 }}>
                  提示：需要对接萤石播放器SDK
                </p>
              </div>
            </div>
          ) : (
            <Empty description="暂无视频" />
          )}
        </Card>

        <Card title="风险分析" extra={<WarningOutlined />}>
          {currentRisk ? (
            <Space direction="vertical" style={{ width: '100%' }} size="large">
              <div>
                <div style={{ marginBottom: 8 }}>
                  <span style={{ fontSize: 18, fontWeight: 'bold' }}>风险分数</span>
                </div>
                <Progress
                  percent={Math.round(currentRisk.risk_score * 100)}
                  strokeColor={
                    currentRisk.risk_level === 'HIGH' ? '#ff4d4f' :
                    currentRisk.risk_level === 'MEDIUM' ? '#faad14' : '#52c41a'
                  }
                  size="small"
                />
              </div>

              <div>
                <div style={{ marginBottom: 8 }}>风险等级</div>
                <Tag color={getRiskColor(currentRisk.risk_level)} style={{ fontSize: 16, padding: '6px 12px' }}>
                  {currentRisk.risk_level}
                </Tag>
              </div>

              <div>
                <div style={{ marginBottom: 8 }}>当前状态</div>
                <div style={{ fontSize: 16 }}>{currentRisk.status}</div>
              </div>

              <div>
                <div style={{ marginBottom: 8 }}>风险原因</div>
                {currentRisk.reasons.map((reason, idx) => (
                  <Tag key={idx} style={{ marginBottom: 4 }}>{reason}</Tag>
                ))}
                {currentRisk.reasons.length === 0 && <span style={{ color: '#999' }}>无</span>}
              </div>

              <div>
                <div style={{ marginBottom: 8, color: '#999', fontSize: 14 }}>
                  最后更新: {new Date(currentRisk.recorded_at).toLocaleString()}
                </div>
              </div>
            </Space>
          ) : (
            <Empty description="暂无风险数据" />
          )}
        </Card>
      </div>
    </div>
  )
}
