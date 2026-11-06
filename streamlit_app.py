"""Local chat UI. AiRi owns conversation logic; main.py owns the workflow."""

import contextlib
import json
import os
import shutil
import sqlite3
import tempfile
import weakref
from pathlib import Path

import streamlit as st

from airi import DEMO_DB_NAME, DEMO_DIR_PREFIX, AiRi, demo_mode
from api import AgentError
from memory import MemoryStore
from persona import build_persona, load_profile
from tools import default_tools
from ui_components import (apply_styles, hero, overview, section_heading, system_state,
                           welcome, workflow_overview)
from usage import DEMO_REQUEST_LIMIT, NORMAL_REQUEST_LIMIT, UsageGuard, UsageLimitReached

# Replies from these commands are shown with st.text: line breaks stay, and text that
# came from the model (/suggest, /pending) is never rendered as Markdown or HTML.
LOCAL_COMMANDS = ("/help", "/memories", "/remember", "/forget", "/pending", "/approve", "/deny",
                  "/suggest", "/usage")
PROFILE_WARNING_TH = "ไม่ได้ใช้ local profile เพราะรูปแบบหรือค่าไม่ถูกต้อง"
DEMO_BANNER_TH = (f"LIVE API DEMO · ส่งข้อความไปยัง OpenAI API จริงและอาจมีค่าใช้จ่าย · "
                  "Memory ชั่วคราวแยกจากของจริง (ลบเมื่อระบบเก็บกวาด session หรือปิด server "
                  "ตามปกติ อาจไม่ลบทันทีที่ปิดแท็บ) · ไม่ใช้ profile ส่วนตัว · "
                  f"เพดาน {DEMO_REQUEST_LIMIT} requests ต่อ session (ไม่ใช่เพดานค่าใช้จ่าย)")
TRACE_LABEL = "รายละเอียดขั้นตอนของ Agents"
CALCULATION_PROMPT = "ช่วยคำนวณ 12 * 7 ให้หน่อย"
OFFLINE_DEMO_NOTICE = (
    "ตอนนี้เป็น Demo Mode ที่ยังไม่ได้ตั้ง API key จึงยังคุยกับ AI, ใช้ /work หรือ /suggest ไม่ได้\n"
    "ลองสาธิตคำนวณ 12 × 7 ด้วยเครื่องมือในเครื่อง หรือใช้คำสั่ง Memory ชั่วคราวได้ "
    "หากต้องการใช้ AI ให้ตั้ง OPENAI_API_KEY ที่เครื่องซึ่งรันแอป แล้วส่งข้อความอีกครั้ง")
OFFLINE_DEMO_BANNER_TH = (
    "OFFLINE DEMO · ยังไม่ได้ตั้ง API key จึงไม่ส่งคำขอ AI · "
    "สาธิตคำนวณในเครื่องและใช้ Memory ชั่วคราวได้ · "
    "ไม่ใช้ Memory หรือ profile ส่วนตัว · Memory ชั่วคราวลบเมื่อระบบเก็บกวาด session "
    "หรือปิด server ตามปกติ อาจไม่ลบทันทีที่ปิดแท็บ")


def status_line(event):
    """Short progress text for one workflow event; work content stays out of it."""
    kind, agent = event["type"], event.get("agent")
    if kind == "agent_finished" and agent in ("Planner", "Researcher"):
        return f"{agent} ✓"
    if kind == "agent_started" and agent == "Builder":
        return f"Builder draft {event['draft']}…"
    if kind == "agent_finished" and agent == "Reviewer":
        return f"Reviewer: {event['status']}"
    if kind == "revision":
        return f"FAIL → แก้ไขรอบ {event['revision']}/{event['max_revisions']}"
    if kind == "finished":
        return f"เสร็จสิ้น · {event['review']} · แก้ไข {event['retries']}/{event['max_retries']} รอบ"
    return None


def render_trace(trace):
    """Status summary first; full agent outputs stay collapsed below it."""
    lines = [line for line in map(status_line, trace) if line]
    st.caption(" · ".join(lines))
    with st.expander(TRACE_LABEL, expanded=False):
        for event in trace:
            if event["type"] != "agent_finished":
                continue
            agent = event["agent"]
            if agent == "Reviewer":
                # status is the validated PASS/FAIL enum, so this label stays system text.
                st.markdown(f"**Reviewer · draft {event['draft']}: {event['status']}**")
                model_code(event["feedback"])
            else:
                title = f"Builder · draft {event['draft']}" if agent == "Builder" else agent
                st.markdown(f"**{title}**")
                model_code(event["content"])


def model_code(body):
    """Model-written work output: shown verbatim, so links, images and HTML never render."""
    st.code(body, language="markdown", wrap_lines=True)


def render_message(message):
    """Render stored messages identically on submission and later reruns."""
    with st.chat_message(message["role"], avatar="🌿" if message["role"] == "assistant" else None):
        st.caption("Local demo · calculate" if message.get("local_tool") else
                   "Demo mode" if message.get("demo_notice") else
                   "AiRi" if message["role"] == "assistant" else "คุณ")
        if message.get("error"):
            st.error(message["content"])
        elif "work" in message:
            outcome = message["work"]
            if outcome["review"] == "PASS":
                st.success("PASS · ผ่านการตรวจจาก Reviewer")
            else:
                st.warning("FAIL · ครบจำนวนแก้ไขแล้ว ควรตรวจผลงานด้วยตัวเอง")
            st.caption(f"Builder แก้ไข {outcome['retries']} รอบ · ผลลัพธ์หลังจบ workflow")
            st.markdown("**ผลงาน**")
            model_code(outcome["result"])
            with st.expander("Reviewer feedback", expanded=True):
                model_code(outcome["feedback"])
            if message.get("trace"):
                render_trace(message["trace"])
        elif message.get("plain"):
            st.text(message["content"], wrap=True)
        elif message["role"] == "assistant":
            # AiRi's chat reply is model text: plain text, never Markdown or HTML.
            st.text(message["content"], wrap=True)
        else:
            st.markdown(message["content"])


def run_work(assistant, task):
    """Run /work with live status; the trace lives only in this session's messages."""
    trace = []
    with st.status("กำลังทำ workflow…", expanded=True) as status:
        def on_event(event):
            trace.append(event)
            line = status_line(event)
            if line:
                status.write(line)

        try:
            outcome = assistant.work(task, on_event=on_event)
        except Exception:
            status.update(label="workflow หยุดทำงาน", state="error")
            raise
        status.update(label="workflow เสร็จสิ้น", state="complete", expanded=False)
    return {"role": "assistant", "work": outcome, "trace": trace}


def demo_without_key():
    """Use the actual isolated session, not the edition label or a changed environment flag."""
    return "demo_cleanup" in st.session_state and not os.getenv("OPENAI_API_KEY", "").strip()


def offline_demo_reply(assistant, message):
    """One explicit existing tool example; no intent inference or fabricated AI replies."""
    if message.strip() in (CALCULATION_PROMPT, "ลองคำนวณ 12 × 7"):
        names = assistant.tools.names() if assistant.tools is not None else ()
        if "calculate" in names:
            result = json.loads(assistant.tools.run("calculate", json.dumps({"expression": "12 * 7"})))
            value = result.get("result", {}).get("value")
            if result.get("ok") and isinstance(value, str):
                return {"role": "assistant", "plain": True, "local_tool": True,
                        "content": f"12 × 7 = {value}\n"
                                   "ผลจากเครื่องมือ calculate ที่คำนวณจริงในเครื่อง · ไม่ได้เรียก AI หรือ API"}
        return {"role": "assistant", "plain": True, "demo_notice": True,
                "content": "session นี้ยังสาธิตเครื่องมือ calculate ไม่ได้ · ไม่ได้เรียก AI หรือ API\n"
                           + OFFLINE_DEMO_NOTICE}
    return {"role": "assistant", "plain": True, "demo_notice": True,
            "content": OFFLINE_DEMO_NOTICE}


def respond(assistant, message):
    """Use AiRi's shared entry points; never create another API or agent loop."""
    command, _, task = message.strip().partition(" ")
    # Both chat input and queued starter callbacks pass here. Check again at dispatch
    # time so a removed key cannot fall through to a previously cached API client.
    if demo_without_key() and (not message.strip().startswith("/")
                               or command == "/work" or message.strip() == "/suggest"):
        return offline_demo_reply(assistant, message)
    if command == "/work":
        return run_work(assistant, task.strip())
    reply = {"role": "assistant", "content": assistant.handle(message)}
    if command in LOCAL_COMMANDS:
        reply["plain"] = True
    return reply


def answer(assistant, message):
    """Return a reply message; errors become safe text without raw exception details."""
    try:
        return respond(assistant, message)
    except UsageLimitReached as error:
        # Checked before AgentError: nothing was sent, so this is not a connection problem.
        return {"role": "assistant", "error": True,
                "content": f"ถึงเพดานการใช้ API ของ session นี้ · {error}"}
    except AgentError as error:
        return {"role": "assistant", "error": True,
                "content": f"ติดต่อ AI ไม่สำเร็จ · {error} ลองส่งข้อความใหม่เมื่อแก้ไขแล้ว"}
    except ValueError as error:
        return {"role": "assistant", "error": True,
                "content": f"คำสั่งหรือข้อความไม่ถูกต้อง · {error}"}
    except (sqlite3.Error, OSError):
        return {"role": "assistant", "error": True,
                "content": "เข้าถึง Memory ไม่ได้ กรุณาตรวจสิทธิ์ local_data แล้วลองใหม่"}
    except Exception:
        # Never expose raw exceptions, paths, or credentials in the UI.
        return {"role": "assistant", "error": True,
                "content": "เกิดข้อผิดพลาดที่ไม่คาดคิด ลองส่งใหม่หรือเริ่มบทสนทนาใหม่"}


def decide(command, action_id):
    """Button callback: send the same /approve or /deny text a user could type."""
    message = f"{command} {action_id}"
    st.session_state.messages.append({"role": "user", "content": message})
    st.session_state.messages.append(answer(st.session_state.airi, message))


def usage_caption(snapshot):
    """Request and token counts for this browser session; not money."""
    text = (f"การใช้ API ใน session นี้: {snapshot['requests']}/{snapshot['limit']} requests · "
            f"input {snapshot['input_tokens']:,} / output {snapshot['output_tokens']:,} tokens")
    if snapshot["missing_usage"]:
        text += f" · ไม่มีข้อมูล usage {snapshot['missing_usage']} ครั้ง"
    if snapshot["failed"]:
        text += f" · ล้มเหลว {snapshot['failed']} ครั้ง"
    return text


def render_usage(assistant, workspace=None):
    """Drawn last, after this run's request, so the numbers are current."""
    if assistant.usage is None:
        return
    caption = usage_caption(assistant.usage.snapshot())
    st.sidebar.caption(caption)
    st.sidebar.caption("เพดานนับจำนวน request ต่อ session นี้เท่านั้น ไม่ใช่เพดานค่าใช้จ่าย")
    if workspace is not None:
        with workspace.container():
            st.caption(caption)


def start_demo_session():
    """Live API demo: temporary memory for this browser session only, and no profile.

    The real Memory DB and the private profile are never opened. The directory is removed
    when this session's AiRi is garbage-collected or when the server process exits.
    """
    directory = tempfile.mkdtemp(prefix=DEMO_DIR_PREFIX)
    try:
        assistant = AiRi(MemoryStore(Path(directory) / DEMO_DB_NAME), persona=build_persona(),
                         tools=default_tools(), usage=UsageGuard(DEMO_REQUEST_LIMIT))
    except (sqlite3.Error, OSError):
        shutil.rmtree(directory, True)
        st.error("เปิด Memory ชั่วคราวของโหมดสาธิตไม่ได้ ลองโหลดหน้าใหม่")
        st.stop()
    st.session_state.demo_cleanup = weakref.finalize(assistant, shutil.rmtree, directory, True)
    st.session_state.profile_warning = False
    st.session_state.airi = assistant


def render_pending(assistant):
    """Show actions waiting for approval. Summaries hold ids and metadata, not note text."""
    for action in assistant.gate.pending():
        if action.kind == "memory.save":
            # The summary holds model-written content and reason: plain text only.
            st.warning(f"รออนุมัติ {action.id} · AiRi เสนอให้จำ (ข้อความด้านล่างแสดงเป็น plain text)")
            st.text(action.summary)
        else:
            st.warning(f"รออนุมัติ {action.id} · {action.summary}")
        approve, deny = st.columns(2)
        approve.button("อนุมัติ", key=f"approve-{action.id}", type="primary",
                       on_click=decide, args=("/approve", action.id))
        deny.button("ปฏิเสธ", key=f"deny-{action.id}",
                    on_click=decide, args=("/deny", action.id))


def render_sidebar():
    """Keep the existing privacy disclosures and session reset within easy reach."""
    with st.sidebar:
        st.subheader("AiRi / Session")
        if st.button("เริ่มบทสนทนาใหม่", key="new-conversation", width="stretch",
                     help="ล้างแชตและรายการรออนุมัติ โดยยังเก็บ Memory และตัวนับ API ไว้"):
            st.session_state.messages = []
            st.session_state.pop("ui_command", None)
            st.session_state.airi.new_conversation()
        st.markdown(
            "**คำสั่ง**\n\n"
            "- `/remember <ข้อความ>` บันทึกสิ่งที่อยากให้จำ\n"
            "- `/memories` ดู Memory ที่บันทึกไว้\n"
            "- `/work <งาน>` เรียก workflow เดิม\n"
            "- `/help` ดูคำสั่งเพิ่มเติม"
        )
        st.caption("Planner → Researcher → Builder → Reviewer พร้อม feedback/retry")
        st.divider()
        if "demo_cleanup" in st.session_state:
            st.caption("โหมดสาธิต: Memory เป็นไฟล์ชั่วคราวเฉพาะ session นี้ แยกจากแท็บอื่น "
                       "ไม่ใช้ฐานข้อมูล Memory ส่วนตัว · ลบเมื่อระบบเก็บกวาด session นี้หรือเมื่อปิด "
                       "server ตามปกติ อาจไม่ลบทันทีที่ปิดแท็บ")
        else:
            st.caption("Memory ใช้ SQLite ในเครื่องร่วมกับ CLI และทุกแท็บของ UI")
        st.caption("ไม่บันทึกแชตอัตโนมัติ · ประวัติในหน้านี้อยู่เฉพาะ session")
        st.caption("แชตส่งข้อความ ประวัติล่าสุด Memory ที่ตรงคำค้น วันเวลาปัจจุบัน "
                   "และสรุปผล /work ล่าสุดที่กรองแล้วไปยัง OpenAI")
        st.caption("/work ส่งโจทย์เข้า Core โดยไม่แนบ Memory หรือประวัติแชต")
        if os.getenv("OPENAI_API_KEY", "").strip():
            st.caption("API: ตั้งค่า environment แล้ว (ยังไม่ได้ตรวจสิทธิ์หรือ quota)")
        else:
            st.info("ยังไม่ได้ตั้ง OPENAI_API_KEY · ใช้คำสั่ง Memory ได้ ส่วนแชตและ /work ต้องใช้ key")
        if "demo_cleanup" in st.session_state:
            if demo_without_key():
                st.info(OFFLINE_DEMO_BANNER_TH)
            else:
                st.warning(DEMO_BANNER_TH)
        if st.session_state.get("profile_warning"):
            st.warning(PROFILE_WARNING_TH)


def queue_command(command):
    """An explicit button click follows the same command path as typing in chat."""
    st.session_state.ui_command = command


def render_controls(assistant, *, api_configured):
    section_heading("01", "Your workspace", "เครื่องมือเล็ก ๆ สำหรับสิ่งที่คุณอยากทำ")
    left, right = st.columns(2, gap="small")
    left.button("ดู Memory", key="show-memories", on_click=queue_command,
                args=("/memories",), width="stretch")
    right.button("คำสั่งทั้งหมด", key="show-help", on_click=queue_command,
                 args=("/help",), width="stretch")
    left.button("รายการรออนุมัติ", key="show-pending", on_click=queue_command,
                args=("/pending",), width="stretch")
    right.button("การใช้ API", key="show-usage", on_click=queue_command,
                 args=("/usage",), width="stretch")
    with st.expander("Memory · สิ่งที่คุณเลือกให้จำ"):
        st.caption("บันทึกเมื่อคุณสั่งเท่านั้น · การลบและสิ่งที่ AI เสนอต้องผ่านการอนุมัติ")
        with st.form("remember-note", clear_on_submit=True):
            note = st.text_area("อยากให้ AiRi จำอะไร", max_chars=2000, key="memory-note",
                                placeholder="เช่น ชอบเรียน Python จากตัวอย่างสั้น ๆ")
            if st.form_submit_button("บันทึก Memory", width="stretch"):
                queue_command(f"/remember {note}")
        st.button("ให้ AiRi เสนอ Memory", key="suggest-memory", on_click=queue_command,
                  args=("/suggest",), disabled=not assistant.history or not api_configured,
                  help="ใช้ข้อความแชตล่าสุดและเรียก API · ทุกข้อเสนอรออนุมัติก่อนบันทึก",
                  width="stretch")
        st.caption("ลบรายการด้วย /forget <id> ในแชต แล้วตรวจการ์ดอนุมัติก่อนตัดสินใจ")
    st.divider()
    section_heading("02", "A thoughtful workflow", "ภาพรวม 4 บทบาท · เริ่มด้วย /work")
    workflow_overview()
    st.caption("Planner → Researcher → Builder → Reviewer · ใช้ /work <งาน> เพื่อเริ่ม "
               "แก้ได้สูงสุด 3 รอบ · PASS เป็นการตรวจของ AI")
    with st.expander("Tools · คำนวณในเครื่อง"):
        names = assistant.tools.names() if assistant.tools is not None else ()
        if not names:
            st.caption("session นี้ไม่ได้เปิด Tool Calling")
        for name in names:
            st.text({"calculate": "calculate · เลขคณิต", "date_math": "date_math · คำนวณวันที่"}.get(name, name))
        st.caption("Demo ไม่มี key: ปุ่มคำนวณสาธิตเรียก calculate ในเครื่อง · ไม่ได้ใช้ AI"
                   if demo_without_key() else
                   "AiRi เลือกเรียกระหว่างแชต · ไม่อ่านไฟล์ ไม่ต่อเครือข่าย และไม่แก้ Memory")
    with st.expander("Session & privacy"):
        usage_view = st.empty()
        st.caption("ไม่บันทึกแชตลงไฟล์ · ข้อมูลที่เกี่ยวข้องถูกส่งไป OpenAI เมื่อใช้ AI")
        st.caption("ดูรายละเอียดโหมด ข้อมูลที่ส่ง และเริ่มบทสนทนาใหม่ได้ในแถบด้านข้าง")
    return usage_view


def dashboard_snapshot(assistant):
    """Read safe metadata via existing interfaces; never render saved note content here."""
    try:
        memory_count = len(assistant.memory.list_memories())
    except (sqlite3.Error, OSError):
        memory_count = None
    return {"memory_count": memory_count, "pending_count": len(assistant.gate.pending()),
            "tool_names": assistant.tools.names() if assistant.tools is not None else (),
            "usage": assistant.usage.snapshot() if assistant.usage is not None else None}


def main():
    st.set_page_config(page_title="AiRi | VINFREY", page_icon="🌿", layout="wide",
                       initial_sidebar_state="collapsed")
    apply_styles()

    if "airi" not in st.session_state:
        if demo_mode():
            start_demo_session()
        else:
            profile, warning = load_profile()
            st.session_state.profile_warning = bool(warning)
            try:
                st.session_state.airi = AiRi(
                    MemoryStore(), persona=build_persona(profile.get("user_name")),
                    tools=default_tools(), usage=UsageGuard(NORMAL_REQUEST_LIMIT))
            except (sqlite3.Error, OSError):
                hero(("Error", "เปิด Memory ไม่สำเร็จ", "error"))
                st.error("เปิด Memory ไม่ได้ กรุณาตรวจสิทธิ์เขียนโฟลเดอร์ local_data แล้วลองโหลดหน้าใหม่")
                st.stop()
    if "messages" not in st.session_state:
        st.session_state.messages = []
    assistant = st.session_state.airi
    render_sidebar()
    api_configured = bool(os.getenv("OPENAI_API_KEY", "").strip())
    hero_slot, overview_slot = st.empty(), st.empty()

    def refresh_dashboard(*, busy=False):
        snapshot = dashboard_snapshot(assistant)
        state = system_state(api_configured=api_configured,
                             memory_available=snapshot["memory_count"] is not None,
                             messages=st.session_state.messages, usage=snapshot["usage"], busy=busy,
                             offline_demo=demo_without_key())
        with hero_slot.container():
            hero(state, live_demo="demo_cleanup" in st.session_state and api_configured,
                 offline_demo=demo_without_key())
        with overview_slot.container():
            overview(**snapshot)

    refresh_dashboard()
    with st.container(key="workspace"):
        controls, conversation = st.columns([.32, .68], gap="large")
        with controls, st.container(key="controls"):
            usage_view = render_controls(assistant, api_configured=api_configured)
        with conversation, st.container(key="conversation"):
            section_heading("03", "A conversation with AiRi", "พื้นที่ของคุณ · คุยต่อได้ใน session เดิม")
            history = st.container(height=380, border=False, key="conversation-history")
            with history:
                if not st.session_state.messages:
                    welcome()
                    with st.container(key="starters"):
                        offline = demo_without_key()
                        local_calculate = assistant.tools is not None and "calculate" in assistant.tools.names()
                        st.caption("Demo ไม่มี key · ปุ่มคำนวณใช้เครื่องมือในเครื่อง ส่วนคำถาม AI จะแจ้งข้อจำกัด"
                                   if offline else "ลองเริ่มจากเรื่องใกล้ตัว · ปุ่มคำถามใช้ API")
                        a, b = st.columns(2)
                        a.button("ช่วยวางแผนการเรียน", key="starter-plan", on_click=queue_command,
                                 args=("ช่วยวางแผนการเรียนสำหรับสัปดาห์นี้",),
                                 disabled=not api_configured and not offline, width="stretch")
                        b.button("ลองคำนวณ 12 × 7", key="starter-calculate", on_click=queue_command,
                                 args=(CALCULATION_PROMPT,),
                                 disabled=not api_configured and not (offline and local_calculate),
                                 help="Demo ไม่มี key: คำนวณจริงในเครื่อง · โหมดอื่นใช้ AI ตามปกติ",
                                 width="stretch")
                for message in st.session_state.messages:
                    render_message(message)
            prompt = st.chat_input("เล่าให้ฟัง หรือพิมพ์ /work ตามด้วยงานที่อยากทำ",
                                   key="conversation-input")
            queued = st.session_state.pop("ui_command", None)
            message = prompt or queued
            if message and message.strip():
                user_message = {"role": "user", "content": message}
                st.session_state.messages.append(user_message)
                refresh_dashboard(busy=True)
                with history:
                    render_message(user_message)
                    is_work = message.strip().partition(" ")[0] == "/work"
                    with contextlib.nullcontext() if is_work else st.spinner(
                            "กำลังตรวจคำขอสาธิต…" if demo_without_key() else "AiRi กำลังตอบ…"):
                        reply = answer(assistant, message)
                    st.session_state.messages.append(reply)
                    render_message(reply)
                refresh_dashboard()
            with st.container(key="conversation-footer"):
                st.caption("คุณเป็นคนตัดสินใจ · Memory ไม่บันทึกอัตโนมัติ · ตรวจงานสำคัญด้วยตัวเองเสมอ")
            with st.container(key="pending"):
                render_pending(assistant)
    render_usage(assistant, usage_view)


if __name__ == "__main__":
    main()
