"""Local chat UI. AiRi owns conversation logic; main.py owns the workflow."""

import contextlib
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
            st.text(message["content"])
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


def respond(assistant, message):
    """Use AiRi's shared entry points; never create another API or agent loop."""
    command, _, task = message.strip().partition(" ")
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


def render_usage(assistant):
    """Drawn last, after this run's request, so the numbers are current."""
    if assistant.usage is None:
        return
    st.sidebar.caption(usage_caption(assistant.usage.snapshot()))
    st.sidebar.caption("เพดานนับจำนวน request ต่อ session นี้เท่านั้น ไม่ใช่เพดานค่าใช้จ่าย")


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


def main():
    st.set_page_config(page_title="AiRi | VINFREY", page_icon="🌿", layout="centered")
    st.title("AiRi 🌿")
    st.caption("VINFREY · AI conversation & reviewed work · Local portfolio demo")

    with st.sidebar:
        st.subheader("คุยกับ AiRi")
        st.write("คุยกันได้ตามปกติ หรือส่งงานให้ workflow ช่วยวางแผนและตรวจผลงาน")
        st.markdown(
            "**คำสั่ง**\n\n"
            "- `/remember <ข้อความ>` บันทึกสิ่งที่อยากให้จำ\n"
            "- `/memories` ดู Memory ที่บันทึกไว้\n"
            "- `/work <งาน>` เรียก workflow เดิม\n"
            "- `/help` ดูคำสั่งเพิ่มเติม"
        )
        st.caption("Planner → Researcher → Builder → Reviewer พร้อม feedback/retry")
        st.divider()
        if demo_mode():
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
                st.error("เปิด Memory ไม่ได้ กรุณาตรวจสิทธิ์เขียนโฟลเดอร์ local_data แล้วลองโหลดหน้าใหม่")
                st.stop()
    if "demo_cleanup" in st.session_state:
        st.sidebar.warning(DEMO_BANNER_TH)
    if st.session_state.get("profile_warning"):
        st.sidebar.warning(PROFILE_WARNING_TH)
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if st.sidebar.button("เริ่มบทสนทนาใหม่", help="ล้างแชตใน session นี้ โดยยังเก็บ Memory ไว้"):
        st.session_state.messages = []
        st.session_state.airi.new_conversation()

    if not st.session_state.messages:
        st.info("วันนี้อยากคุยเรื่องอะไร หรือมีงานให้ช่วยคิดก็ส่งมาได้")
    for message in st.session_state.messages:
        render_message(message)

    prompt = st.chat_input("พิมพ์ข้อความ หรือ /work ตามด้วยงานที่อยากทำ")
    if prompt and prompt.strip():
        user_message = {"role": "user", "content": prompt}
        st.session_state.messages.append(user_message)
        render_message(user_message)
        # /work shows its own st.status progress; chat keeps the simple spinner.
        is_work = prompt.strip().partition(" ")[0] == "/work"
        with contextlib.nullcontext() if is_work else st.spinner("AiRi กำลังตอบ…"):
            reply = answer(st.session_state.airi, prompt)
        st.session_state.messages.append(reply)
        render_message(reply)

    # After handling input so a new /forget request shows its card on the same run.
    render_pending(st.session_state.airi)
    render_usage(st.session_state.airi)


if __name__ == "__main__":
    main()
