import { useState, useEffect } from 'react'
import { Card, Table, Tag, Space, Badge } from 'antd'
import { CheckCircleOutlined, CloseCircleOutlined } from '@ant-design/icons'
import { deviceApi } from '../services/api'
import type { Device } from '../types'

export default function DeviceStatus() {
  const [devices, setDevices] = useState<Device[]>([])
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    loadDevices()
  }, [])

  const loadDevices = async () => {
    setLoading(true)
    try {
      const res = await deviceApi.getAll()
      setDevices(res.data)
    } catch (error) {
      console.error('加载设备失败:', error)
    } finally {
      setLoading(false)
    }
  }

  const columns = [
    {
      title: '设备ID',
      dataIndex: 'id',
      key: 'id',
    },
    {
      title: '设备类型',
      dataIndex: 'device_type',
      key: 'device_type',
      render: (type: string) => (
        <Tag color={type === 'camera' ? 'blue' : 'purple'}>
          {type === 'camera' ? '摄像机' : '雷达'}
        </Tag>
      ),
    },
    {
      title: '设备名称',
      dataIndex: 'device_name',
      key: 'device_name',
      render: (name: string) => name || '-',
    },
    {
      title: '在线状态',
      dataIndex: 'online_status',
      key: 'online_status',
      render: (online: boolean) => (
        <Space>
          <Badge status={online ? 'success' : 'error'} />
          <span>{online ? '在线' : '离线'}</span>
        </Space>
      ),
    },
    {
      title: '设备序列号',
      dataIndex: 'device_serial',
      key: 'device_serial',
      render: (serial: string) => {
        // 脱敏展示：显示前4位和后4位
        if (serial.length > 12) {
          return `${serial.slice(0, 4)}****${serial.slice(-4)}`
        }
        return serial
      },
    },
    {
      title: '位置',
      dataIndex: 'location',
      key: 'location',
      render: (loc: string) => loc || '-',
    },
    {
      title: '最后在线',
      dataIndex: 'last_seen_at',
      key: 'last_seen_at',
      render: (time: string) => time ? new Date(time).toLocaleString() : '-',
    },
  ]

  const onlineCount = devices.filter(d => d.online_status).length
  const cameraCount = devices.filter(d => d.device_type === 'camera').length
  const radarCount = devices.filter(d => d.device_type === 'radar').length

  return (
    <div>
      <h1>设备状态</h1>

      <Space size="large" style={{ marginBottom: 24 }}>
        <Card size="small">
          <div>总设备数: <strong>{devices.length}</strong></div>
        </Card>
        <Card size="small">
          <div>在线设备: <strong style={{ color: '#52c41a' }}>{onlineCount}</strong></div>
        </Card>
        <Card size="small">
          <div>摄像机: <strong>{cameraCount}</strong></div>
        </Card>
        <Card size="small">
          <div>雷达: <strong>{radarCount}</strong></div>
        </Card>
      </Space>

      <Table
        columns={columns}
        dataSource={devices}
        rowKey="id"
        loading={loading}
      />
    </div>
  )
}
