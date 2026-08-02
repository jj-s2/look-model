-- 添加告警升级相关字段
ALTER TABLE alerts ADD COLUMN IF NOT EXISTS escalation_level INTEGER DEFAULT 0;
ALTER TABLE alerts ADD COLUMN IF NOT EXISTS escalated_at TIMESTAMP WITH TIME ZONE;

-- 修改状态字段长度以支持新状态
ALTER TABLE alerts ALTER COLUMN status TYPE VARCHAR(30);

-- 创建系统日志表
CREATE TABLE IF NOT EXISTS system_logs (
    id SERIAL PRIMARY KEY,
    source VARCHAR(50) NOT NULL,
    level VARCHAR(10) NOT NULL,
    message TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 创建索引
CREATE INDEX IF NOT EXISTS idx_logs_source_level ON system_logs(source, level);
CREATE INDEX IF NOT EXISTS idx_logs_created_at ON system_logs(created_at);
CREATE INDEX IF NOT EXISTS idx_logs_source ON system_logs(source);
CREATE INDEX IF NOT EXISTS idx_logs_level ON system_logs(level);

-- 添加注释
COMMENT ON TABLE system_logs IS '系统日志表';
COMMENT ON COLUMN system_logs.source IS '日志来源: EZVIZ_SDK, AI_ENGINE, WEBSOCKET, USER_ACTION, SYSTEM';
COMMENT ON COLUMN system_logs.level IS '日志级别: DEBUG, INFO, WARN, ERROR, CRITICAL';
COMMENT ON COLUMN alerts.escalation_level IS '升级级别: 0-正常, 1-首次升级, 2-二次升级';
COMMENT ON COLUMN alerts.escalated_at IS '升级时间';
