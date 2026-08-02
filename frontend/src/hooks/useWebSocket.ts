import { useEffect, useRef, useState } from 'react'
import type { WebSocketMessage } from '../types'

export function useWebSocket(clientId: string = 'dashboard') {
  const [connected, setConnected] = useState(false)
  const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null)
  const wsRef = useRef<WebSocket | null>(null)
  const pingIntervalRef = useRef<NodeJS.Timeout | null>(null)
  const pongTimeoutRef = useRef<NodeJS.Timeout | null>(null)
  const lastPongTimeRef = useRef<number>(Date.now())

  useEffect(() => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const wsUrl = `${protocol}//${window.location.host}/ws/${clientId}`

    const connect = () => {
      const ws = new WebSocket(wsUrl)

      ws.onopen = () => {
        console.log('✅ WebSocket连接成功')
        setConnected(true)
        lastPongTimeRef.current = Date.now()
        
        // 启动心跳机制
        startHeartbeat(ws)
      }

      ws.onmessage = (event) => {
        // 处理pong响应
        if (event.data === 'pong') {
          lastPongTimeRef.current = Date.now()
          return
        }

        try {
          const message: WebSocketMessage = JSON.parse(event.data)
          setLastMessage(message)
        } catch (error) {
          console.error('解析WebSocket消息失败:', error)
        }
      }

      ws.onerror = (error) => {
        console.error('❌ WebSocket错误:', error)
      }

      ws.onclose = () => {
        console.log('❌ WebSocket断开，5秒后重连...')
        setConnected(false)
        stopHeartbeat()
        setTimeout(connect, 5000)
      }

      wsRef.current = ws
    }

    const startHeartbeat = (ws: WebSocket) => {
      // 每30秒发送一次ping
      pingIntervalRef.current = setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) {
          ws.send('ping')
          console.log('🏓 发送心跳 ping')
        }
      }, 30000)

      // 每10秒检查一次是否收到pong
      pongTimeoutRef.current = setInterval(() => {
        const timeSinceLastPong = Date.now() - lastPongTimeRef.current
        if (timeSinceLastPong > 60000) {
          // 超过60秒未收到pong，认为连接已断开
          console.warn('⚠️ 心跳超时，强制重连')
          ws.close()
        }
      }, 10000)
    }

    const stopHeartbeat = () => {
      if (pingIntervalRef.current) {
        clearInterval(pingIntervalRef.current)
        pingIntervalRef.current = null
      }
      if (pongTimeoutRef.current) {
        clearInterval(pongTimeoutRef.current)
        pongTimeoutRef.current = null
      }
    }

    connect()

    return () => {
      stopHeartbeat()
      if (wsRef.current) {
        wsRef.current.close()
      }
    }
  }, [clientId])

  const send = (message: any) => {
    if (wsRef.current && connected) {
      wsRef.current.send(JSON.stringify(message))
    }
  }

  return { connected, lastMessage, send }
}
