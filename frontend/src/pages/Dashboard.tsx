import { useEffect, useState } from 'react'
import { Card, Row, Col, Statistic, Alert as AntAlert, Spin } from 'antd'
import { UserOutlined, SafetyOutlined, BellOutlined, HeartOutlined } from '@ant-design/icons'
import { deviceApi, riskApi, alertApi, healthApi } from '../services/api'
import { useWebSocket } from '../hooks/useWebSocket'

export default function Dashboard() {
  const [loading, setLoading] = useState(true)
  const [backendHealthy, setBackendHealthy] = useState(false)
  const [stats, setStats] = useState({
    onlineDevices: 0,
    totalDevices: 0,
    currentRiskLevel: 'LOW' as 'LOW' | 'MEDIUM' | 'HIGH',
    todayAlerts: 0,
    heartRate: 0,
  })

  const { connected, lastMessage } = useWebSocket('dashboard')

  useEffect(() => {
    loadData()
  }, [])

  useEffect(() => {
    if (lastMessage) {
      // 实时更新数据
      if (lastMessage.type === 'risk_update') {
        loadData()
      } else if (lastMessage.type === 'alert') {
        setStats(prev => ({ ...prev, todayAlerts: prev.todayAlerts + 1 }))
      }
    }
  }, [lastMessage])

  const loadData = async () => {
    try {
      const [devicesRes, risksRes, alertsRes, healthRes] = await Promise.all([
        deviceApi.getAll(),
        riskApi.getCurrent(),
        alertApi.getAll({ limit: 100 }),
        healthApi.getLatest(),
      ])

      const onlineDevices = devicesRes.data.filter(d => d.online_status).length
      const maxRisk = Math.max(...risksRes.data.map(r => r.risk_score), 0)
      const riskLevel = maxRisk >= 0.8 ? 'HIGH' : maxRisk >= 0.6 ? 'MEDIUM' : 'LOW'
      const todayAlerts = alertsRes.data.filter(a => {
        const today = new Date().toDateString()
        return new Date(a.created_at).toDateString() === today
      }).length

      const latestHealth = healthRes.data[0]

      setStats({
        onlineDevices,
        totalDevices: devicesRes.data.length,
        currentRiskLevel: riskLevel,
        todayAlerts,
        heartRate: latestHealth?.heart_rate || 0,
      })
      setBackendHealthy(true)
    } catch (error) {
      console.error('加载数据失败:', error)
      setBackendHealthy(false)
    } finally {
      setLoading(false)
    }
  }

  const getRiskColor = () => {
    if (stats.currentRiskLevel === 'HIGH') return '#ff4d4f'
    if (stats.currentRiskLevel === 'MEDIUM') return '#faad14'
    return '#52c41a'
  }

  if (loading) {
    return <div style={{ textAlign: 'center', padding: 50 }}><Spin size="large" /></div>
  }

  return (
    <div>
      <h1>系统概览</h1>
      
      {!connected && (
        <AntAlert
          message="WebSocket未连接"
          description="实时数据推送不可用，请检查网络连接"
          type="warning"
          showIcon
          style={{ marginBottom: 24 }}
        />
      )}

      <Row gutter={[24, 24]}>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="在线设备"
              value={stats.onlineDevices}
              suffix={`/ ${stats.totalDevices}`}
              prefix={<UserOutlined />}
              valueStyle={{ color: '#3f8600' }}
            />
          </Card>
        </Col>

        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="当前风险等级"
              value={stats.currentRiskLevel}
              prefix={<SafetyOutlined />}
              valueStyle={{ color: getRiskColor(), fontSize: 28 }}
            />
          </Card>
        </Col>

        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="今日告警"
              value={stats.todayAlerts}
              prefix={<BellOutlined />}
              valueStyle={{ color: stats.todayAlerts > 0 ? '#cf1322' : '#3f8600' }}
            />
          </Card>
        </Col>

        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="最新心率"
              value={stats.heartRate}
              suffix="bpm"
              prefix={<HeartOutlined />}
              valueStyle={{ color: '#1890ff' }}
            />
          </Card>
        </Col>
      </Row>

      <Card title="系统状态" style={{ marginTop: 24 }}>
        <p style={{ fontSize: 16 }}>
          WebSocket: <span style={{ color: connected ? '#52c41a' : '#ff4d4f' }}>
            {connected ? '已连接' : '未连接'}
          </span>
        </p>
        <p style={{ fontSize: 16 }}>
          后端服务: <span style={{ color: backendHealthy ? '#52c41a' : '#ff4d4f' }}>
            {backendHealthy ? '正常' : '异常'}
          </span>
        </p>
      </Card>
    </div>
  )
}
