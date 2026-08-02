import { useState, useEffect } from 'react'
import { Table, Tag, Button, Space, Modal, Input, message, Select } from 'antd'
import { CheckOutlined, CloseOutlined, WarningOutlined } from '@ant-design/icons'
import { alertApi } from '../services/api'
import { useWebSocket } from '../hooks/useWebSocket'
import type { Alert } from '../types'

export default function AlertCenter() {
  const [alerts, setAlerts] = useState<Alert[]>([])
  const [loading, setLoading] = useState(false)
  const [statusFilter, setStatusFilter] = useState<string>('')
  const [modalVisible, setModalVisible] = useState(false)
  const [modalType, setModalType] = useState<'confirm' | 'resolve' | 'false'>('confirm')
  const [currentAlert, setCurrentAlert] = useState<Alert | null>(null)
  const [handler, setHandler] = useState('')
  const [note, setNote] = useState('')
  const { lastMessage } = useWebSocket('alerts')

  useEffect(() => {
    loadAlerts()
  }, [statusFilter])

  useEffect(() => {
    if (lastMessage?.type === 'alert') {
      loadAlerts()
      message.warning(`新告警: ${lastMessage.data.message}`)
    }
  }, [lastMessage])

  const loadAlerts = async () => {
    setLoading(true)
    try {
      const res = await alertApi.getAll({ 
        status_filter: statusFilter || undefined,
        limit: 100 
      })
      setAlerts(res.data)
    } catch (error) {
      console.error('加载告警失败:', error)
      message.error('加载告警失败')
    } finally {
      setLoading(false)
    }
  }

  const handleAction = (alert: Alert, type: 'confirm' | 'resolve' | 'false') => {
    setCurrentAlert(alert)
    setModalType(type)
    setModalVisible(true)
    setHandler('')
    setNote('')
  }

  const submitAction = async () => {
    if (!currentAlert || !handler) {
      message.error('请填写处理人')
      return
    }

    try {
      if (modalType === 'confirm') {
        await alertApi.confirm(currentAlert.id, { handler, note })
        message.success('告警已确认')
      } else if (modalType === 'resolve') {
        if (!note) {
          message.error('请填写处理说明')
          return
        }
        await alertApi.resolve(currentAlert.id, { handler, note })
        message.success('告警已处理')
      } else {
        await alertApi.markFalseAlarm(currentAlert.id, handler)
        message.success('已标记为误报')
      }
      setModalVisible(false)
      loadAlerts()
    } catch (error) {
      console.error('操作失败:', error)
      message.error('操作失败')
    }
  }

  const columns = [
    {
      title: 'ID',
      dataIndex: 'id',
      key: 'id',
      width: 60,
    },
    {
      title: '设备',
      dataIndex: 'device_id',
      key: 'device_id',
    },
    {
      title: '风险等级',
      dataIndex: 'risk_level',
      key: 'risk_level',
      render: (level: string) => (
        <Tag color={level === 'HIGH' ? 'red' : level === 'MEDIUM' ? 'orange' : 'green'}>
          {level}
        </Tag>
      ),
    },
    {
      title: '告警信息',
      dataIndex: 'message',
      key: 'message',
      ellipsis: true,
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => {
        const colorMap: Record<string, string> = {
          NEW: 'red',
          NOTIFIED: 'orange',
          CONFIRMED: 'blue',
          RESOLVED: 'green',
          FALSE_ALARM: 'gray',
        }
        return <Tag color={colorMap[status] || 'default'}>{status}</Tag>
      },
    },
    {
      title: '创建时间',
      dataIndex: 'created_at',
      key: 'created_at',
      render: (time: string) => new Date(time).toLocaleString(),
    },
    {
      title: '操作',
      key: 'actions',
      render: (_: any, record: Alert) => (
        <Space>
          {['NEW', 'NOTIFIED'].includes(record.status) && (
            <Button size="small" icon={<CheckOutlined />} onClick={() => handleAction(record, 'confirm')}>
              确认
            </Button>
          )}
          {['CONFIRMED'].includes(record.status) && (
            <Button size="small" type="primary" onClick={() => handleAction(record, 'resolve')}>
              处理完成
            </Button>
          )}
          {!['RESOLVED', 'FALSE_ALARM'].includes(record.status) && (
            <Button size="small" danger icon={<CloseOutlined />} onClick={() => handleAction(record, 'false')}>
              误报
            </Button>
          )}
        </Space>
      ),
    },
  ]

  return (
    <div>
      <h1>告警中心</h1>

      <div style={{ marginBottom: 24 }}>
        <Space>
          <span>状态筛选：</span>
          <Select
            style={{ width: 150 }}
            value={statusFilter}
            onChange={setStatusFilter}
            options={[
              { label: '全部', value: '' },
              { label: 'NEW', value: 'NEW' },
              { label: 'CONFIRMED', value: 'CONFIRMED' },
              { label: 'RESOLVED', value: 'RESOLVED' },
              { label: 'FALSE_ALARM', value: 'FALSE_ALARM' },
            ]}
          />
          <Button onClick={loadAlerts}>刷新</Button>
        </Space>
      </div>

      <Table
        columns={columns}
        dataSource={alerts}
        rowKey="id"
        loading={loading}
        pagination={{ pageSize: 20 }}
      />

      <Modal
        title={
          modalType === 'confirm' ? '确认告警' :
          modalType === 'resolve' ? '处理告警' : '标记误报'
        }
        open={modalVisible}
        onOk={submitAction}
        onCancel={() => setModalVisible(false)}
      >
        <Space direction="vertical" style={{ width: '100%' }}>
          <Input
            placeholder="处理人"
            value={handler}
            onChange={(e) => setHandler(e.target.value)}
          />
          {modalType !== 'false' && (
            <Input.TextArea
              placeholder={modalType === 'resolve' ? '处理说明（必填）' : '备注（选填）'}
              value={note}
              onChange={(e) => setNote(e.target.value)}
              rows={4}
            />
          )}
        </Space>
      </Modal>
    </div>
  )
}
