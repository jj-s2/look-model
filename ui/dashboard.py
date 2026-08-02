"""Local Gradio dashboard with a dependency-light view-model seam for tests."""

from __future__ import annotations

from typing import Any

from pipeline.live_service import LiveMonitoringService, ServiceSnapshot


SCREENING_NOTICE = "心理健康信息仅用于风险筛查不构成诊断；GDS-15 仅可由用户主动发起。"


def dashboard_view_model(snapshot: ServiceSnapshot) -> dict[str, Any]:
    """Create redaction-safe display data without importing Gradio or a browser."""
    fall_events = [
        {"level": decision.level, "score": decision.score, "evidence": list(decision.reasons)}
        for decision in snapshot.decisions
        if decision.kind == "fall_event"
    ]
    wellbeing_changes = [
        {"level": decision.level, "score": decision.score, "evidence": list(decision.reasons)}
        for decision in snapshot.decisions
        if decision.kind == "wellbeing_change"
    ]
    return {
        "watermark": "演示数据 / demo=true" if snapshot.demo else "",
        "device_quality": {"camera": snapshot.camera_health, "radar": snapshot.radar_health},
        "fall_events": fall_events,
        "fall_trend": [item["score"] for item in fall_events],
        "wellbeing_changes": wellbeing_changes,
        "wellbeing_trend": [item["score"] for item in wellbeing_changes],
        "evidence": [reason for decision in snapshot.decisions for reason in decision.reasons],
        "alert_history": list(snapshot.alert_history),
        "screening_notice": SCREENING_NOTICE,
        "latest_frame": snapshot.latest_frame,
        "errors": list(snapshot.source_errors),
    }


def build_dashboard(service: LiveMonitoringService):
    """Build, but do not launch, the optional local Gradio monitoring interface."""
    try:
        import gradio as gr
    except ModuleNotFoundError as error:
        raise RuntimeError("Gradio 未安装。请安装 requirements.txt 中的 gradio>=5,<7 后再启动界面。") from error

    def refresh():
        model = dashboard_view_model(service.step())
        return (
            model["watermark"], model["device_quality"], model["latest_frame"], model["fall_events"],
            model["fall_trend"], model["wellbeing_changes"], model["wellbeing_trend"], model["evidence"],
            model["alert_history"], model["errors"],
        )

    with gr.Blocks(title="本地老人安全监测") as dashboard:
        watermark = gr.Markdown(visible=True)
        gr.Markdown("# 本地老人安全监测\n仅展示经本地处理的最小必要信息。")
        with gr.Row():
            device_quality = gr.JSON(label="设备在线与质量")
            latest_frame = gr.Image(label="实时画面", type="numpy")
        fall_events = gr.JSON(label="跌倒事件")
        fall_trend = gr.JSON(label="跌倒趋势")
        gr.Markdown("## 心理健康变化趋势\n" + SCREENING_NOTICE)
        wellbeing_changes = gr.JSON(label="心理变化")
        wellbeing_trend = gr.JSON(label="心理趋势")
        evidence = gr.JSON(label="证据说明")
        alert_history = gr.JSON(label="告警历史")
        errors = gr.JSON(label="组件状态")
        gr.Button("我主动发起 GDS-15 筛查", interactive=True)
        refresh_button = gr.Button("刷新本地状态")
        refresh_button.click(
            refresh,
            outputs=[watermark, device_quality, latest_frame, fall_events, fall_trend, wellbeing_changes,
                     wellbeing_trend, evidence, alert_history, errors],
        )
    return dashboard
