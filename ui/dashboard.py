"""Local Gradio dashboard with a dependency-light view-model seam for tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from mental.gds15 import GDS15
from pipeline.live_service import LiveMonitoringService, ServiceSnapshot


SCREENING_NOTICE = "心理健康信息仅用于风险筛查不构成诊断；GDS-15 仅可由用户主动发起。"
GDS15_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "screening" / "gds15_zh.json"


def load_gds15() -> GDS15:
    """Load the reviewed local questionnaire; this never contacts its source URL."""
    return GDS15.from_json(GDS15_CONFIG)


def start_gds15_screening(gds: GDS15 | None = None) -> dict[str, Any]:
    """Expose GDS-15 only after the person has actively requested screening."""
    scale = gds or load_gds15()
    return {
        "active": True,
        "questions": [
            {"id": item.id, "question_zh": item.question_zh, "options": item.options}
            for item in scale.items
        ],
        "notice": SCREENING_NOTICE,
    }


def submit_gds15_answers(answers: Mapping[str, bool], gds: GDS15 | None = None) -> dict[str, Any]:
    """Score a voluntarily completed GDS-15 response set without diagnostic labels."""
    result = (gds or load_gds15()).score(answers)
    return {
        "active": False,
        "score": result.score,
        "risk_band": result.risk_band,
        "is_diagnosis": result.is_diagnosis,
        "notice": SCREENING_NOTICE,
    }


def dashboard_view_model(snapshot: ServiceSnapshot) -> dict[str, Any]:
    """Create redaction-safe display data without importing Gradio or a browser."""
    state_labels = {
        "invite_candidate": "建议自愿简短问候",
        "screening_concern": "筛查关注（非诊断）",
        "abstained": "证据不足，暂不判断",
        "human_review_required": "需要人工复核",
    }

    def display(decision):
        item = {"level": decision.level, "score": decision.score, "evidence": list(decision.reasons)}
        if decision.kind == "wellbeing_change":
            state = decision.state or next((reason for reason in decision.reasons if reason in state_labels), None)
            item.update({
                "state_zh": state_labels.get(state or "", "心理变化提示"),
                "delivery_scope": decision.delivery_scope,
                "uncertainty": decision.uncertainty,
                "is_diagnosis": False,
            })
        return item

    fall_events = [
        display(decision)
        for decision in snapshot.decisions
        if decision.kind == "fall_event"
    ]
    prefall_warnings = [display(decision) for decision in snapshot.decisions if decision.kind == "prefall_warning"]
    fall_forecasts = [display(decision) for decision in snapshot.decisions if decision.kind == "fall_forecast"]
    wellbeing_changes = [
        display(decision)
        for decision in snapshot.decisions
        if decision.kind == "wellbeing_change"
    ]
    return {
        "watermark": "演示数据 / demo=true" if snapshot.demo else "",
        "device_quality": {"camera": snapshot.camera_health, "radar": snapshot.radar_health},
        "fall_events": fall_events,
        "emergency_events": [item for item, decision in zip(fall_events, (d for d in snapshot.decisions if d.kind == "fall_event")) if decision.level == "critical"],
        "prefall_warnings": prefall_warnings,
        "fall_forecasts": fall_forecasts,
        "fall_trend": [item["score"] for item in fall_events],
        "prefall_trend": [item["score"] for item in prefall_warnings],
        "wellbeing_changes": wellbeing_changes,
        "wellbeing_prompts": wellbeing_changes,
        "wellbeing_trend": [item["score"] for item in wellbeing_changes],
        "wellbeing_prompt": snapshot.wellbeing_prompt,
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

    scale = load_gds15()

    def refresh():
        model = dashboard_view_model(service.step())
        return (
            model["watermark"], model["device_quality"], model["latest_frame"], model["emergency_events"],
            model["prefall_warnings"], model["fall_forecasts"], model["fall_trend"], model["wellbeing_prompts"], model["wellbeing_trend"], model["evidence"],
            model["alert_history"], model["errors"], model["wellbeing_prompt"],
        )

    def begin_gds15():
        payload = start_gds15_screening(scale)
        return gr.update(visible=payload["active"]), "请自主完成全部 15 题后提交。" + payload["notice"]

    def submit_gds15(*selected_options: str | None):
        if len(selected_options) != len(scale.items) or any(option is None for option in selected_options):
            return "请先完整回答 15 题。" + SCREENING_NOTICE
        answers = {
            item.id: selected == item.options[0]
            for item, selected in zip(scale.items, selected_options)
        }
        result = submit_gds15_answers(answers, scale)
        return f"筛查分数：{result['score']}；风险分层：{result['risk_band']}。{result['notice']}"

    with gr.Blocks(title="本地老人安全监测") as dashboard:
        watermark = gr.Markdown(visible=True)
        gr.Markdown("# 本地老人安全监测\n仅展示经本地处理的最小必要信息。")
        with gr.Row():
            device_quality = gr.JSON(label="设备在线与质量")
            latest_frame = gr.Image(label="实时画面", type="numpy")
        fall_events = gr.JSON(label="已确认跌倒事件")
        prefall_warnings = gr.JSON(label="跌倒前预警")
        fall_forecasts = gr.JSON(label="跌倒风险预测")
        fall_trend = gr.JSON(label="跌倒趋势")
        gr.Markdown("## 心理健康变化趋势\n" + SCREENING_NOTICE)
        wellbeing_changes = gr.JSON(label="心理变化提示")
        wellbeing_trend = gr.JSON(label="心理趋势")
        evidence = gr.JSON(label="证据说明")
        alert_history = gr.JSON(label="告警历史")
        errors = gr.JSON(label="组件状态")
        wellbeing_prompt = gr.JSON(label="自愿短问候（本地）")
        gds_start = gr.Button("我主动发起 GDS-15 筛查", interactive=True)
        gds_status = gr.Markdown(SCREENING_NOTICE)
        with gr.Column(visible=False) as gds_panel:
            gds_answers = [
                gr.Radio(choices=list(item.options), label=item.question_zh, type="value")
                for item in scale.items
            ]
            gds_submit = gr.Button("提交 GDS-15 筛查", variant="primary")
        refresh_button = gr.Button("刷新本地状态")
        refresh_button.click(
            refresh,
            outputs=[watermark, device_quality, latest_frame, fall_events, prefall_warnings, fall_forecasts,
                     fall_trend, wellbeing_changes, wellbeing_trend, evidence, alert_history, errors, wellbeing_prompt],
        )
        gds_start.click(begin_gds15, outputs=[gds_panel, gds_status])
        gds_submit.click(submit_gds15, inputs=gds_answers, outputs=gds_status)
    return dashboard
