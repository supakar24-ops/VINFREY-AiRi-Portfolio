"""Presentation-only components. Never pass model, memory, or profile text to HTML."""

from html import escape
from pathlib import Path

import streamlit as st

APP_EDITION = "Portfolio Demo"
STYLE_PATH = Path(__file__).with_name("ui.css")


def apply_styles():
    st.html(f"<style>{STYLE_PATH.read_text(encoding='utf-8')}</style>")


def system_state(*, api_configured, memory_available, messages, usage=None, busy=False,
                 offline_demo=False):
    """Describe observed local state; a configured key is not a successful API check."""
    if busy:
        return "Processing", "กำลังประมวลผลคำขอของคุณ", "busy"
    if not memory_available:
        return "Memory unavailable", "เข้าถึง Memory ไม่ได้ · ลองใหม่อีกครั้ง", "error"
    if usage is not None and usage["remaining"] <= 0:
        return "Limit reached", "ถึงเพดาน API ของ session · คำสั่งในเครื่องยังใช้ได้", "warning"
    if messages and messages[-1].get("error"):
        return "Error", "คำขอล่าสุดไม่สำเร็จ · ดูรายละเอียดในบทสนทนา", "error"
    if not api_configured:
        if offline_demo:
            return "Offline demo", "สาธิตคำนวณและ Memory ชั่วคราวได้ · AI ต้องใช้ API key", "warning"
        return "Local controls", "ยังไม่ได้ตั้ง API key · ใช้คำสั่ง Memory ได้", "warning"
    # A successful conversational reply/workflow is evidence; a local command is not.
    if any(m["role"] == "assistant" and not m.get("error")
           and ("work" in m or ("content" in m and not m.get("plain"))) for m in messages):
        return "Ready", "คำขอ AI ก่อนหน้าในบทสนทนานี้สำเร็จ", "ready"
    return "Configured", "ตั้งค่า API key แล้ว · ยังไม่ได้ตรวจการเชื่อมต่อหรือ quota", "neutral"


def hero(state, *, live_demo=False, offline_demo=False):
    label, detail, tone = state
    mode = (f"{APP_EDITION} · Offline" if offline_demo else
            f"{APP_EDITION} · Live API" if live_demo else f"{APP_EDITION} · Normal mode")
    st.html(f"""
      <section class="airi-hero" aria-label="AiRi Personal AI Companion">
        <div class="airi-brand">
          <div class="airi-logo" aria-hidden="true"><span></span><i></i></div>
          <div><div class="airi-eyebrow">VINFREY / YOUR EVERYDAY THINKING SPACE</div>
            <div class="airi-wordmark">AiRi<span class="airi-wordmark-dot">.</span></div>
            <div class="airi-subtitle">Personal AI Companion</div>
          </div>
        </div>
        <div class="airi-hero-note">พื้นที่สำหรับคุย คิด และทำงานไปด้วยกัน<br>
          <span>Memory ที่คุณเลือกจำ · Workflow ที่คุณตรวจสอบได้</span></div>
        <div class="airi-hero-status">
          <span class="airi-edition">{escape(mode)}</span>
          <span class="airi-status airi-status-{escape(tone)}"><i></i>{escape(label)}</span>
          <span class="airi-status-detail">{escape(detail)}</span>
        </div>
      </section>""")


def overview(*, memory_count, pending_count, tool_names, usage):
    count = "—" if memory_count is None else str(memory_count)
    memory_detail = "ตรวจสอบไม่ได้" if memory_count is None else "บันทึกในเครื่อง · ไม่แสดงเนื้อหา"
    request_value = f"{usage['requests']} / {usage['limit']}" if usage else "—"
    request_detail = "requests ใน session นี้" if usage else "session นี้ไม่ได้เปิดตัวนับ"
    cards = (
        ("01", "MEMORY", count, memory_detail),
        ("02", "APPROVALS", str(pending_count), "รายการที่รอให้คุณตัดสินใจ"),
        ("03", "CHAT TOOLS", str(len(tool_names)), "เครื่องมือคำนวณที่เปิดใช้งาน"),
        ("04", "API USAGE", request_value, request_detail),
    )
    st.html('<section class="airi-overview" aria-label="System overview">' + "".join(
        f'<div class="airi-stat"><div class="airi-stat-label"><span>{number}</span>{label}</div>'
        f'<div class="airi-stat-value">{escape(value)}</div>'
        f'<div class="airi-stat-detail">{escape(detail)}</div></div>'
        for number, label, value, detail in cards) + '</section>')


def section_heading(number, title, subtitle):
    # Only fixed UI copy is supplied by the application.
    st.html(f'<div class="airi-section-heading"><span>{escape(number)}</span>'
            f'<div><h2>{escape(title)}</h2><p>{escape(subtitle)}</p></div></div>')


def workflow_overview():
    steps = (("Planner", "วางแผน"), ("Researcher", "เตรียมข้อมูล"),
             ("Builder", "สร้างผลงาน"), ("Reviewer", "ตรวจและให้ feedback"))
    st.html('<div class="airi-workflow">' + "".join(
        f'<div class="airi-step"><span>{i:02}</span><div><b>{name}</b><small>{detail}</small>'
        '</div></div>' for i, (name, detail) in enumerate(steps, 1)) + '</div>')


def welcome():
    # UI greeting, deliberately outside the transcript and outbound model history.
    st.html('''<div class="airi-welcome"><div class="airi-welcome-mark" aria-hidden="true">a.</div>
      <div class="airi-eyebrow">A LITTLE CLARITY STARTS HERE</div>
      <h3>สวัสดี เราคือ AiRi</h3>
      <p>มีเรื่องอยากคุย หรือโจทย์ที่อยากช่วยคิด<br>ค่อย ๆ เริ่มจากสิ่งที่อยู่ในใจตอนนี้ก็ได้</p>
      <div class="airi-welcome-tags"><span>คุยและต่อยอดความคิด</span>
        <span>จำสิ่งที่คุณเลือก</span><span>ช่วยวางแผนและตรวจงาน</span></div>
      </div>''')
