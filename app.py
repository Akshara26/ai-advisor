import streamlit as st
import uuid
from advisor.graph import chat
from advisor.memory import load_history, save_history, delete_history
import io
import re
from pypdf import PdfReader
from advisor.transcript_utils import parse_transcript


st.title("UMN CS Graduate Advisor")
st.caption("Ask me anything about the CS graduate program at University of Minnesota.")
if "transcript_upload_key" not in st.session_state:
    st.session_state.transcript_upload_key = str(uuid.uuid4())

# ── Sidebar: Transcript → Degree Audit ───────────────────────────────────────
with st.sidebar:
    st.header("📄 Degree Audit")
    st.caption(
    "The PDF is not stored. Course codes, grades, earned credits, and cumulative GPA "
    "are extracted in memory. Selected course information is included in your conversation history, which is kept for 7 days."
    )

    uploaded = st.file_uploader(
        "Transcript PDF",
        type=["pdf"],
        label_visibility="collapsed",
        key=st.session_state.transcript_upload_key,
    )

    if uploaded:
        try:
            result = parse_transcript(uploaded.read())
            all_courses  = result["courses"]
            course_records = result["course_records"]
            csci_courses = [c for c in all_courses if c.startswith("CSCI")]
            other_courses = [c for c in all_courses if not c.startswith("CSCI")]

            if not csci_courses:
                st.warning("No completed CSCI courses found in this transcript.")
            else:
                st.success(
                    f"Found {len(csci_courses)} CSCI courses"
                    + (f" and {len(other_courses)} others" if other_courses else "")
                )

                selected_csci = st.multiselect(
                    "CSCI courses:",
                    options=csci_courses,
                    default=csci_courses,
                )
                other_default = ", ".join(other_courses) if other_courses else ""
                other_raw = st.text_input(
                    "Other courses (optional, comma-separated):",
                    value=other_default,
                    placeholder="e.g. STAT5302, MOT5001",
                )
                selected_other = (
                    [c.strip().upper().replace(" ", "") for c in other_raw.split(",") if c.strip()]
                    if other_raw else []
                )

                program = st.selectbox("Program:", ["MS", "PhD"])

                plan = None
                if program == "MS":
                    plan = st.selectbox("M.S. Plan:", ["A", "B", "C"])

                if result["gpa"]:
                    st.metric("Cumulative GPA", result["gpa"])

                if st.button("Run degree audit →", use_container_width=True, type="primary"):
                    record_by_code = {
                        record["code"]: record
                        for record in course_records
                    }
                    selected = selected_csci + selected_other

                    selected_with_credits = []

                    for code in selected:
                        record = record_by_code.get(code)

                        if record:
                            selected_with_credits.append(
                                f"{code} ({record['credits']:g} credits)"
                            )
                        else:
                            selected_with_credits.append(code)
                    if program == "MS":
                        program_text = f"M.S. Plan {plan}"
                    else:
                        program_text = "Ph.D."

                    audit_q = (
                        f"I'm in the CSCI {program_text} program. "
                        f"I have completed: {', '.join(selected_with_credits)}. "
                        f"What requirements do I still need to fulfill to graduate?"
                    )
                    st.session_state.pending_question = audit_q
                    st.rerun()

        except Exception as e:
            st.error(f"Could not parse transcript: {e}")
            st.caption("Try downloading a fresh copy from MyU → Academics → Unofficial Transcript.")

# Persist session ID across refreshes using query params
params = st.query_params
if "session_id" not in params:
    new_id = str(uuid.uuid4())
    st.query_params["session_id"] = new_id

session_id = st.query_params["session_id"]

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = load_history(session_id)

if "messages" not in st.session_state:
    st.session_state.messages = list(st.session_state.conversation_history)

# Clear saved history and start a fresh session
with st.sidebar:
    if st.button("Clear conversation", use_container_width=True):
        try:
            delete_history(session_id)
        except Exception:
            st.error("Could not delete saved history. Please try again.")
        else:
            st.session_state.messages = []
            st.session_state.conversation_history = []
            st.session_state.pop("pending_question", None)
            st.session_state.transcript_upload_key = str(uuid.uuid4())
            st.query_params["session_id"] = str(uuid.uuid4())
            st.rerun()

def process_message(prompt):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response, st.session_state.conversation_history, drafted_email, _ = chat(
                prompt, st.session_state.conversation_history
            )
        st.markdown(response)
        if drafted_email:
            st.divider()
            st.caption("Draft email")
            st.text_area("Draft email", value=drafted_email, height=200,
                         key=f"email_{len(st.session_state.messages)}")
    st.session_state.messages.append({"role": "assistant", "content": response})
    save_history(session_id, st.session_state.conversation_history)

# Display existing messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle question injected from transcript audit
if "pending_question" in st.session_state:
    pending = st.session_state.pop("pending_question")
    process_message(pending)

# Handle typed input
if prompt := st.chat_input("Ask a question..."):
    process_message(prompt)