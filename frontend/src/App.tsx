import { Routes, Route, Navigate } from 'react-router-dom'
import { Layout, Menu } from 'antd'
import {
  DashboardOutlined,
  VideoCameraOutlined,
  LineChartOutlined,
  BellOutlined,
  ClusterOutlined,
} from '@ant-design/icons'
import { useNavigate, useLocation } from 'react-router-dom'
import Dashboard from './pages/Dashboard'
import LiveMonitor from './pages/LiveMonitor'
import RiskTrend from './pages/RiskTrend'
import AlertCenter from './pages/AlertCenter'
import DeviceStatus from './pages/DeviceStatus'

const { Header, Content, Sider } = Layout

function App() {
  const navigate = useNavigate()
  const location = useLocation()

  const menuItems = [
    { key: '/dashboard', icon: <DashboardOutlined />, label: '概览' },
    { key: '/live', icon: <VideoCameraOutlined />, label: '实时监控' },
    { key: '/trends', icon: <LineChartOutlined />, label: '风险趋势' },
    { key: '/alerts', icon: <BellOutlined />, label: '告警中心' },
    { key: '/devices', icon: <ClusterOutlined />, label: '设备状态' },
  ]

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Header style={{ display: 'flex', alignItems: 'center', padding: '0 24px' }}>
        <div style={{ color: 'white', fontSize: '20px', fontWeight: 'bold' }}>
          老年人多模态AI监测预警平台
        </div>
      </Header>
      <Layout>
        <Sider width={200} theme="light">
          <Menu
            mode="inline"
            selectedKeys={[location.pathname]}
            style={{ height: '100%', borderRight: 0 }}
            items={menuItems}
            onClick={({ key }) => navigate(key)}
          />
        </Sider>
        <Layout style={{ padding: '24px' }}>
          <Content
            style={{
              padding: 24,
              margin: 0,
              minHeight: 280,
              background: '#fff',
              borderRadius: 8,
            }}
          >
            <Routes>
              <Route path="/" element={<Navigate to="/dashboard" replace />} />
              <Route path="/dashboard" element={<Dashboard />} />
              <Route path="/live" element={<LiveMonitor />} />
              <Route path="/trends" element={<RiskTrend />} />
              <Route path="/alerts" element={<AlertCenter />} />
              <Route path="/devices" element={<DeviceStatus />} />
            </Routes>
          </Content>
        </Layout>
      </Layout>
    </Layout>
  )
}

export default App
