import { useState, useEffect } from 'react'
import { Card, Select, Space, Empty } from 'antd'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts'
import { deviceApi, riskApi } from '../services/api'
import type { Device, RiskRecord } from '../types'

export default function RiskTrend() {
  const [devices, setDevices] = useState<Device[]>([])
  const [selectedDevice, setSelectedDevice] = useState<string>('')
  const [period, setPeriod] = useState<number>(24)
  const [chartData, setChartData] = useState<any[]>([])

  useEffect(() => {
    loadDevices()
  }, [])

  useEffect(() => {
    if (selectedDevice) {
      loadRiskHistory()
    }
  }, [selectedDevice, period])

  const loadDevices = async () => {
    try {
      const res = await deviceApi.getAll()
      setDevices(res.data)
      if (res.data.length > 0) {
        setSelectedDevice(res.data[0].id)
      }
    } catch (error) {
      console.error('加载设备失败:', error)
    }
  }

  const loadRiskHistory = async () => {
    try {
      const res = await riskApi.getHistory({ device_id: selectedDevice, hours: period })
      const data = res.data
        .sort((a, b) => new Date(a.recorded_at).getTime() - new Date(b.recorded_at).getTime())
        .map(r => ({
          time: new Date(r.recorded_at).toLocaleTimeString('zh-CN', { 
            month: '2-digit', 
            day: '2-digit', 
            hour: '2-digit', 
            minute: '2-digit' 
          }),
          risk_score: r.risk_score,
          risk_level: r.risk_level,
        }))
      setChartData(data)
    } catch (error) {
      console.error('加载风险历史失败:', error)
    }
  }

  return (
    <div>
      <h1>风险趋势</h1>

      <Card style={{ marginBottom: 24 }}>
        <Space size="large">
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

          <span>时间范围：</span>
          <Select
            style={{ width: 150 }}
            value={period}
            onChange={setPeriod}
            options={[
              { label: '24小时', value: 24 },
              { label: '7天', value: 168 },
              { label: '30天', value: 720 },
            ]}
          />
        </Space>
      </Card>

      <Card title="风险分数趋势">
        {chartData.length > 0 ? (
          <ResponsiveContainer width="100%" height={400}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                dataKey="time" 
                tick={{ fontSize: 12 }}
                angle={-45}
                textAnchor="end"
                height={80}
              />
              <YAxis domain={[0, 1]} />
              <Tooltip />
              <Legend />
              <Line 
                type="monotone" 
                dataKey="risk_score" 
                stroke="#ff4d4f" 
                name="风险分数"
                strokeWidth={2}
                dot={{ r: 3 }}
              />
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <Empty description="暂无数据" />
        )}
      </Card>

      <Card title="统计信息" style={{ marginTop: 24 }}>
        {chartData.length > 0 ? (
          <Space size="large">
            <div>
              <div style={{ fontSize: 14, color: '#999' }}>平均风险分数</div>
              <div style={{ fontSize: 24, fontWeight: 'bold' }}>
                {(chartData.reduce((sum, d) => sum + d.risk_score, 0) / chartData.length).toFixed(2)}
              </div>
            </div>
            <div>
              <div style={{ fontSize: 14, color: '#999' }}>最高风险分数</div>
              <div style={{ fontSize: 24, fontWeight: 'bold', color: '#ff4d4f' }}>
                {Math.max(...chartData.map(d => d.risk_score)).toFixed(2)}
              </div>
            </div>
            <div>
              <div style={{ fontSize: 14, color: '#999' }}>数据点数</div>
              <div style={{ fontSize: 24, fontWeight: 'bold' }}>
                {chartData.length}
              </div>
            </div>
          </Space>
        ) : (
          <Empty description="暂无统计数据" />
        )}
      </Card>
    </div>
  )
}
