import streamlit as st
import uuid
from graph import chat
from memory import load_history, save_history
import io
import re
from pypdf import PdfReader

def parse_transcript(pdf_bytes: bytes) -> dict:
    """
    Parse a UMN unofficial transcript PDF.
    Returns completed course codes (e.g. CSCI5523) and cumulative GPA.
    Only includes courses with earned credits > 0 and a valid grade.
    """
    reader = PdfReader(io.BytesIO(pdf_bytes))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)

    courses, gpa = [], None

    for line in text.split("\n"):
        line = line.strip()
        if not line or "TERM GPA" in line or "TERM TOTALS" in line:
            continue

        cum_match = re.search(r"CUM GPA:\s*([\d.]+)", line)
        if cum_match:
            gpa = cum_match.group(1)

        dept_match = re.match(r"^([A-Z]{2,5})\s+(\d{4})\b", line)
        if not dept_match:
            continue

        dept, num = dept_match.group(1), dept_match.group(2)

        # Completed courses have: ... earned grade points at end of line
        grade_match = re.search(
            r"(\d+\.\d+)\s+(\d+\.\d+)\s+([A-Z][+-]?|S|U)\s+\d+\.\d+\s*$",
            line
        )
        if grade_match:
            _, earned, _ = grade_match.groups()
            if float(earned) > 0:
                courses.append(f"{dept}{num}")

    return {"courses": courses, "gpa": gpa}

st.title("UMN CS Graduate Advisor")
st.caption("Ask me anything about the CS graduate program at University of Minnesota.")

# ── Sidebar: Transcript → Degree Audit ───────────────────────────────────────
with st.sidebar:
    st.header("📄 Degree Audit")
    st.caption("Upload your UMN unofficial transcript to skip typing your courses.")

    uploaded = st.file_uploader(
        "Transcript PDF", type=["pdf"], label_visibility="collapsed"
    )

    if uploaded:
        try:
            result = parse_transcript(uploaded.read())
            all_courses  = result["courses"]
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

                if result["gpa"]:
                    st.metric("Cumulative GPA", result["gpa"])

                if st.button("Run degree audit →", use_container_width=True, type="primary"):
                    selected = selected_csci + selected_other
                    gpa_note = f" My cumulative GPA is {result['gpa']}." if result["gpa"] else ""
                    audit_q = (
                        f"I'm in the CSCI {program} program. "
                        f"I have completed: {', '.join(selected)}.{gpa_note} "
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

if "messages" not in st.session_state:
    st.session_state.messages = []

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = load_history(session_id)

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
            st.caption("I couldn't fully answer this from the handbook. Here's a draft email to the graduate coordinators:")
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