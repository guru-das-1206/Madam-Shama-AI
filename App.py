import streamlit as st
import streamlit.components.v1 as components
from google import genai
from google.genai import types
import json, os, re, io, time, hashlib, hmac
from datetime import datetime, date, timedelta
from gtts import gTTS
import pandas as pd
import PyPDF2

# ==========================================
# 1. PAGE + STYLE
# ==========================================
st.set_page_config(page_title="Madam Shama AI", page_icon="👩‍🏫", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600&display=swap');
html, body, [class*="css"], button, input, textarea, select { font-family: 'Poppins', sans-serif; }
.stApp { background: linear-gradient(135deg, #090e17 0%, #111827 100%); color: #f8fafc; }
.stChatMessage { border-radius: 16px; padding: 14px 18px; margin-bottom: 14px; border: 1px solid rgba(255,255,255,0.05); }
.stChatMessage[data-testid="stChatMessageUser"] { background: rgba(30,41,59,.85); border-right: 4px solid #10b981; }
.stChatMessage[data-testid="stChatMessageAssistant"] { background: rgba(15,23,42,.85); border-left: 4px solid #8b5cf6; }
.stChatMessage p, .stChatMessage li { color: #f8fafc !important; font-size: 1.05rem; line-height: 1.6; }
[data-testid="stSidebar"] { background-color: #0f172a; border-right: 1px solid #1e293b; }
.meet-room { background: linear-gradient(180deg,#1e293b,#0f172a); padding: 20px; border-radius: 24px; border: 1px solid #334155; margin-bottom: 16px; }
.video-grid { display: flex; justify-content: center; gap: 20px; flex-wrap: wrap; }
.video-card { background:#1e293b; width: 230px; height: 150px; border-radius: 16px; display:flex; flex-direction:column; align-items:center; justify-content:center; border:2px solid #334155; }
.video-card.active { border: 3px solid #8b5cf6; box-shadow: 0 0 25px rgba(139,92,246,.4); }
.avatar { font-size: 54px; } .name-tag { font-size: 13px; font-weight: 600; }
.feature-card { background: rgba(30,41,59,.6); padding: 18px; border-radius: 16px; border: 1px solid #334155; text-align:center; height:100%; }
.tech-badge { background:#1e293b; padding:6px 12px; border-radius:20px; font-size:14px; color:#8b5cf6; border:1px solid #334155; font-weight:600; display:inline-block; margin-bottom:16px; }
#MainMenu {visibility: hidden;} footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.title("👩‍🏫 Madam Shama: Your AI Coach")
st.markdown('<div class="tech-badge">✨ Powered by Google Gemini · Built for Google Developer Groups (GDG)</div>', unsafe_allow_html=True)

# ==========================================
# 2. API + AI HELPERS
# ==========================================
def secret(name, default=None):
    try: return st.secrets[name]
    except Exception: return default

api_key = secret("GEMINI_API_KEY") or st.sidebar.text_input("Gemini API Key", type="password")

@st.cache_resource
def get_client(key): return genai.Client(api_key=key) if key and key.strip() else None
client = get_client(api_key)
if not client:
    st.info("👈 Enter your Gemini API key in the sidebar to begin.")
    st.stop()

# Verify current model names in Google's Gemini API docs; set MODELS in secrets.toml to override.
MODELS = list(secret("MODELS", ["gemini-2.0-flash", "gemini-2.5-flash"]))

PERSONA = ("You are 'Madam Shama', a caring, patient career and exam coach and tutor. Warm, encouraging, concise. "
           "Help students understand, not just get answers. In practice quizzes and exams never reveal answers before the student attempts. "
           "Keep replies short unless asked for detail. Never invent URLs or facts; say when you are unsure.")

def ask(prompt, system=None, json_mode=False, tools=None, media=None):
    cfg = types.GenerateContentConfig(
        system_instruction=system or PERSONA,
        response_mime_type="application/json" if json_mode else None,
        tools=tools)
    contents = [prompt, types.Part.from_bytes(data=media[0], mime_type=media[1])] if media else prompt
    last = ""
    for m in MODELS:
        try:
            return client.models.generate_content(model=m, contents=contents, config=cfg).text or ""
        except Exception as e:
            last = str(e)
            if any(c in last for c in ("503", "404", "429", "UNAVAILABLE", "NOT_FOUND")): continue
            break
    return f"AI Error: {last}"

def jload(raw):
    try: return json.loads(re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip())
    except Exception: return None

def transcribe(b, mime="audio/wav"):
    t = ask("Transcribe this audio exactly. Return only the text.", system="You are a transcription tool.", media=(b, mime))
    return "" if t.startswith("AI Error") else t.strip()

def play_voice(text):
    clean = re.sub(r"[*#\[\]_>|`~]", "", text).strip()[:700]
    if len(clean) < 3: return None
    try:
        audio = io.BytesIO()
        gTTS(text=clean, lang="en", tld=st.session_state.get("accent", "co.uk (British)").split(" ")[0]).write_to_fp(audio)
        audio.seek(0); return audio
    except Exception: return None

def speak(text):
    if st.session_state.get("tts_on", True):
        a = play_voice(text)
        if a: st.audio(a, format="audio/mp3", autoplay=True)

# ==========================================
# 3. USER DATA (one file per student)
# ==========================================
def blank():
    return {"step": "ask_role", "role": "", "skills": "", "roadmap_data": None, "topics": [], "exam_duration": 60,
            "interview_scores": [], "quiz_scores": [], "mock_scores": [], "topic_stats": {},
            "skill_projects_completed": 0, "done": {}, "activity": [], "last_visit": None}

def U(): return st.session_state.user_data
def pfile(uid): return f"profiles/{uid}.json"

@st.cache_resource
def get_gcp():
    """Connect to Firestore (progress) and Cloud Storage (uploads). Returns (db, bucket, error)."""
    try: info = dict(st.secrets["gcp_service_account"])
    except Exception: info = None
    if info is None and not os.environ.get("K_SERVICE"): return None, None, None   # not configured -> local mode
    try:
        from google.cloud import firestore, storage
        bn = secret("GCS_BUCKET")
        if info:
            from google.oauth2 import service_account
            creds = service_account.Credentials.from_service_account_info(info)
            db = firestore.Client(credentials=creds, project=info["project_id"])
            sc = storage.Client(credentials=creds, project=info["project_id"])
        else:   # running on Google Cloud Run: use the service's own identity
            db = firestore.Client(); sc = storage.Client()
        return db, (sc.bucket(bn) if bn else None), None
    except Exception as e:
        return None, None, str(e)
DB, BUCKET, GCP_ERR = get_gcp()

def load_profile(uid):
    """Saved data, or None for a new student. Raises if the cloud is unreachable (never overwrite with blank)."""
    if DB:
        snap = DB.collection("profiles").document(uid).get()
        return json.loads(snap.to_dict().get("data", "{}")) if snap.exists else None
    try:
        with open(pfile(uid)) as fh: return json.load(fh)
    except FileNotFoundError: return None

def save():
    uid = st.session_state.uid
    try:
        if DB:
            DB.collection("profiles").document(uid).set({"data": json.dumps(U()), "updated": datetime.utcnow().isoformat()})
        else:
            os.makedirs("profiles", exist_ok=True); tmp = pfile(uid) + ".tmp"
            with open(tmp, "w") as fh: json.dump(U(), fh)
            os.replace(tmp, pfile(uid))
    except Exception: st.toast("⚠️ Couldn't save your progress just now.")

def delete_profile(uid):
    try:
        if DB: DB.collection("profiles").document(uid).delete()
        else: os.remove(pfile(uid))
    except Exception: pass

def reset_transient():
    for k in ("quiz", "mock", "iv", "doubt", "proj", "lib_out"): st.session_state.pop(k, None)

def switch_user(uid):
    st.session_state.uid = uid
    data = blank()
    try: saved = load_profile(uid)
    except Exception as e:
        st.session_state.pop("uid", None)
        st.error(f"Couldn't load your saved progress: {e}"); st.stop()
    if saved: data.update(saved)
    st.session_state.user_data = data
    reset_transient()
    try: away = (date.today() - date.fromisoformat(data["last_visit"])).days
    except Exception: away = None
    st.session_state["away"] = away or 0
    if data["step"] == "coach_mode":
        if not away: g = f"Welcome back! Let's continue your prep for **{data['role']}**."
        elif away == 1: g = f"Welcome back! It's been 1 day. Let's keep the streak going for **{data['role']}**. {recommend_next_activity()}"
        else: g = f"Welcome back! It's been **{away} days** since your last practice. Let's ease back into **{data['role']}**. {recommend_next_activity()}"
    else:
        g = "Hello! I'm Madam Shama. Tell me: **what exam or job role are you preparing for?**"
    st.session_state.messages = [{"role": "assistant", "content": g, "is_new": True}]
    data["last_visit"] = date.today().isoformat(); save()

def touch():
    d = date.today().isoformat()
    if d not in U()["activity"]: U()["activity"].append(d)

def streak():
    days, n, d = set(U()["activity"]), 0, date.today()
    if d.isoformat() not in days: d -= timedelta(days=1)
    while d.isoformat() in days: n += 1; d -= timedelta(days=1)
    return n

def rec_topic(topic, ok):
    s = U()["topic_stats"].setdefault(topic or "General", [0, 0])
    s[0] += 1 if ok else 0; s[1] += 1

def weak_topics():
    items = [(t, c / n) for t, (c, n) in U()["topic_stats"].items() if n >= 1 and c / n < 0.7]
    return [t for t, _ in sorted(items, key=lambda x: x[1])]

def avg(lst, k=3): return sum(lst[-k:]) / len(lst[-k:]) if lst else 0

def readiness():
    u = U(); gaps = (u.get("roadmap_data") or {}).get("gaps", [])
    high = sum(1 for g in gaps if g.get("priority") == "High")
    parts = {"Knowledge": (max(0, 30 - high * 8) if u.get("roadmap_data") else 0, 30),
             "Quizzes": (round(avg(u["quiz_scores"]) / 100 * 20), 20),
             "Mock papers": (round(avg(u["mock_scores"]) / 100 * 20), 20),
             "Interviews": (round(avg(u["interview_scores"]) / 10 * 20), 20),
             "Projects": (min(10, u["skill_projects_completed"] * 2), 10)}
    return {"total": min(100, sum(v for v, _ in parts.values())), "parts": parts}

def ctx():
    u = U()
    return (f"Student goal: {u['role']}. Background: {u['skills']}. Weak topics: {', '.join(weak_topics()) or 'none yet'}. "
            f"Days away before this session: {st.session_state.get('away', 0)}. Recent quiz avg: {avg(u['quiz_scores']):.0f}%, mock avg: {avg(u['mock_scores']):.0f}%, interview avg: {avg(u['interview_scores']):.1f}/10.")

def need_setup():
    if U()["step"] != "coach_mode":
        st.warning("⚠️ Please finish setup in the 💬 Coach tab first (tell Madam Shama your goal and skills).")
        return True
    return False

# ==========================================
# 4. AGENT TOOLS (Gemini function calling)
# ==========================================
def get_progress_summary() -> dict:
    """Return the student's readiness score, recent scores and weak topics."""
    return {"readiness": readiness()["total"], "weak_topics": weak_topics(), "quiz_scores": U()["quiz_scores"][-3:],
            "mock_scores": U()["mock_scores"][-3:], "interview_scores": U()["interview_scores"][-3:]}

def add_study_task(task: str, week: int) -> str:
    """Add a task to the student's study plan for the given week number (1 to 4)."""
    rd = U().get("roadmap_data") or {}; U()["roadmap_data"] = rd
    weeks = rd.setdefault("roadmap", [])
    for w in weeks:
        if str(w.get("week")) == str(week):
            w.setdefault("tasks", []).append(task); break
    else: weeks.append({"week": week, "focus": "Extra practice", "tasks": [task]})
    save(); return f"Added to week {week}: {task}"

def recommend_next_activity() -> str:
    """Suggest the single best next practice activity for this student."""
    w = weak_topics()
    if w: return f"Take a Rapid Quiz on '{w[0]}', your weakest topic."
    if not U()["mock_scores"]: return "Attempt a full timed Mock Paper."
    if len(U()["interview_scores"]) < 3: return "Do a Voice Interview round."
    return "Build a project in Skill Builder."

def coach_reply(text):
    hist = "\n".join(f"{m['role']}: {m['content']}" for m in st.session_state.messages[-9:-1])
    prompt = f"{ctx()}\n\nConversation so far:\n{hist}\n\nStudent: {text}"
    if st.session_state.get("use_search"):
        tools = [types.Tool(google_search=types.GoogleSearch())]   # verify in current google-genai docs
    else:
        tools = [get_progress_summary, add_study_task, recommend_next_activity]
    return ask(prompt, tools=tools)

# ==========================================
# 5. VOICE INPUT WIDGETS
# ==========================================
MIC_HTML = """
<div style="font-family:sans-serif;display:flex;gap:8px;align-items:center;flex-wrap:wrap">
<button id="b" style="padding:8px 14px;border-radius:20px;border:1px solid #8b5cf6;background:#1e293b;color:#f8fafc;cursor:pointer">🎤 Start speaking</button>
<button id="sd" style="padding:8px 14px;border-radius:20px;border:1px solid #10b981;background:#064e3b;color:#f8fafc;cursor:pointer">__SENDLABEL__</button>
<span id="m" style="color:#94a3b8;font-size:13px"></span></div>
<script>
const P = window.parent, MODE = '__MODE__', SR = P.SpeechRecognition || P.webkitSpeechRecognition;
const b = document.getElementById('b'), sd = document.getElementById('sd'), m = document.getElementById('m');
if (MODE !== 'chat' && !P.__shamaFocus) {
  P.document.addEventListener('focusin', e => { const t = e.target;
    if (t && t.matches && t.matches('textarea, input[type="text"]')) P.__lastBox = t; });
  P.__shamaFocus = true;
}
function box() {
  if (MODE === 'chat') return P.document.querySelector('textarea[data-testid="stChatInputTextArea"]') || P.document.querySelector('textarea');
  const l = P.__lastBox; return (l && P.document.contains(l)) ? l : null;
}
function setVal(t) {
  const el = box(); if (!el) return;
  const proto = el.tagName === 'INPUT' ? P.HTMLInputElement.prototype : P.HTMLTextAreaElement.prototype;
  Object.getOwnPropertyDescriptor(proto, 'value').set.call(el, t);
  el.dispatchEvent(new P.Event('input', {bubbles: true}));
}
function doSend() {
  const el = box(); if (!el) { m.textContent = 'Tap inside the answer box first.'; return; }
  if (MODE === 'chat') {
    const btn = P.document.querySelector('[data-testid="stChatInputSubmitButton"]');
    if (btn) btn.click();
    else el.dispatchEvent(new P.KeyboardEvent('keydown', {key: 'Enter', code: 'Enter', keyCode: 13, which: 13, bubbles: true}));
  } else { el.blur(); m.textContent = 'Text applied. Now press the main button.'; }
}
if (!SR) { m.textContent = 'Live voice needs Chrome or Edge. Type, or use the recorder.'; b.disabled = true; }
else {
  let rec = null, on = false, base = '', pending = false;
  b.onclick = () => {
    if (on) { rec.stop(); return; }
    const el = box(); if (!el) { m.textContent = 'Tap inside the answer box first, then tap 🎤.'; return; }
    rec = new SR(); rec.lang = '__LANG__'; rec.interimResults = true; rec.continuous = true;
    base = el.value || '';
    rec.onresult = e => { let t = ''; for (let i = 0; i < e.results.length; i++) t += e.results[i][0].transcript;
      setVal((base ? base + ' ' : '') + t); };
    rec.onstart = () => { on = true; b.textContent = '⏹ Stop'; m.textContent = 'Listening… your words appear in the box'; };
    rec.onend = () => { on = false; b.textContent = '🎤 Start speaking'; m.textContent = 'Check the text, fix it if needed, then send.';
      if (pending) { pending = false; setTimeout(doSend, 250); } };
    rec.onerror = e => { m.textContent = 'Mic error: ' + e.error; };
    rec.start();
  };
  sd.onclick = () => { if (on) { pending = true; rec.stop(); } else doSend(); };
}
if (!SR) sd.onclick = doSend;
</script>
"""

def live_mic(mode="chat"):
    """mode='chat': types into the chat box and has a Send button. mode='focus': types into whichever box you last tapped; button applies the text."""
    html = (MIC_HTML.replace("__LANG__", st.session_state.get("lang", "en-IN").split(" ")[0])
            .replace("__MODE__", mode).replace("__SENDLABEL__", "📤 Send" if mode == "chat" else "✅ Done"))
    components.html(html, height=56)

def audio_fallback(key):
    with st.expander("🎙️ Mic not working? Record here instead"):
        a = st.audio_input("Record your message", key=f"{key}_rec")
        if a:
            b = a.getvalue(); h = hash(b)
            if st.session_state.get(f"{key}_h") != h:
                with st.spinner("Transcribing..."):
                    st.session_state[f"{key}_t"] = transcribe(b, a.type or "audio/wav")
                st.session_state[f"{key}_h"] = h
            txt = st.text_area("Edit if needed:", st.session_state.get(f"{key}_t", ""), key=f"{key}_ed_{h}")
            if st.button("✉️ Send", key=f"{key}_send") and txt.strip(): return txt
    return None

def timer(secs):
    components.html("""
    <div id="t" style="font-family:sans-serif;font-size:34px;color:#8b5cf6;font-weight:bold;text-align:center;background:#1e293b;padding:14px;border-radius:15px;border:2px solid #8b5cf6"></div>
    <script>let e = Date.now() + SECS * 1000; const t = document.getElementById('t');
    function tick() { let d = e - Date.now(); if (d <= 0) { t.textContent = '🚨 TIME UP!'; return; }
      t.textContent = '⏳ ' + Math.floor(d / 60000) + 'm ' + Math.floor(d % 60000 / 1000) + 's left'; setTimeout(tick, 500); }
    tick();</script>""".replace("SECS", str(int(secs))), height=90)

# ==========================================
# 6. SIDEBAR
# ==========================================
DEMOS = {
    "👩‍🏫 Demo: Primary Teacher": ("Primary School Teacher", "Good communication", {
        "summary": "Good foundation, needs classroom management focus.", "exam_duration_minutes": 60,
        "topics": ["Child psychology", "Classroom management", "Pedagogy", "Assessment", "Subject knowledge"],
        "gaps": [{"skill": "Classroom Management", "why": "Crucial for student focus.", "priority": "High"}],
        "roadmap": [{"week": 1, "focus": "Basics", "tasks": ["Learn Piaget stages"]}]}),
    "📊 Demo: Data Analyst": ("Data Analyst", "Excel basics", {
        "summary": "Strong Excel, needs SQL and statistics.", "exam_duration_minutes": 60,
        "topics": ["SQL", "Statistics", "Excel", "Data visualization", "Python basics"],
        "gaps": [{"skill": "SQL joins", "why": "Used in every analyst role.", "priority": "High"}],
        "roadmap": [{"week": 1, "focus": "SQL basics", "tasks": ["Practice SELECT and JOIN"]}]}),
}

# ---- Register / Log in (demo-level; see notes about Google Sign-In) ----
USERS_FILE = "profiles/accounts.registry"
def load_users():
    try:
        with open(USERS_FILE) as fh: return json.load(fh)
    except Exception: return {}
def save_users(u):
    os.makedirs("profiles", exist_ok=True)
    with open(USERS_FILE, "w") as fh: json.dump(u, fh)
def hash_pw(pw, salt): return hashlib.pbkdf2_hmac("sha256", pw.encode(), bytes.fromhex(salt), 120000).hex()
def clean_id(x): return re.sub(r"[^a-z0-9_-]", "", x.strip().lower())[:30]

def google_ready():
    try: return hasattr(st, "login") and "auth" in st.secrets
    except Exception: return False
GOOGLE = google_ready()

if GCP_ERR:
    st.error(f"Cloud storage is configured but failed to connect: {GCP_ERR}"); st.stop()

if GOOGLE:
    if not st.user.is_logged_in:
        st.markdown("### 👋 Welcome! Sign in to save your progress")
        st.button("🔐 Sign in with Google", on_click=st.login, type="primary")
        st.stop()
    _email = str(st.user.email).lower()
    _uid = "g" + hashlib.sha256(_email.encode()).hexdigest()[:24]
    st.session_state["display"] = getattr(st.user, "name", None) or _email
    if st.session_state.get("uid") != _uid: switch_user(_uid)
elif "uid" not in st.session_state:
    st.markdown("### 👋 Welcome! Log in or create your free account")
    t_login, t_reg = st.tabs(["🔑 Log in", "📝 Register"])
    with t_login:
        with st.form("login"):
            lu = st.text_input("Username"); lp = st.text_input("Password", type="password")
            if st.form_submit_button("Log in", type="primary"):
                rec = load_users().get(clean_id(lu))
                if rec and hmac.compare_digest(rec["hash"], hash_pw(lp, rec["salt"])): switch_user(clean_id(lu)); st.rerun()
                else: st.error("Wrong username or password.")
    with t_reg:
        with st.form("register"):
            ru = st.text_input("Choose a username (letters, numbers, - _)")
            rp = st.text_input("Password (6+ characters)", type="password"); rp2 = st.text_input("Repeat password", type="password")
            if st.form_submit_button("Create account", type="primary"):
                nid = clean_id(ru); users = load_users()
                if len(nid) < 3: st.error("Username needs at least 3 letters or numbers.")
                elif nid in users: st.error("That username is taken.")
                elif len(rp) < 6 or rp != rp2: st.error("Passwords must match and be 6+ characters.")
                else:
                    salt = os.urandom(16).hex()
                    users[nid] = {"salt": salt, "hash": hash_pw(rp, salt), "created": date.today().isoformat()}
                    save_users(users); switch_user(nid); st.rerun()
    st.stop()

with st.sidebar:
    st.caption(f"Signed in as **{st.session_state.get('display', st.session_state.uid)}**")
    st.caption("☁️ Cloud saving on" if DB else "💾 Local saving only (may reset)")
    if st.button("🚪 Log out", use_container_width=True):
        for k in ("uid", "user_data", "messages", "away", "display"): st.session_state.pop(k, None)
        reset_transient()
        if GOOGLE: st.logout()
        st.rerun()
    if st.button("🔄 Reset my profile", use_container_width=True):
        delete_profile(st.session_state.uid)
        switch_user(st.session_state.uid); st.rerun()
    st.divider()
    st.markdown("### 🚀 Instant Demo Profiles")
    for label, (role, skills, rd) in DEMOS.items():
        if st.button(label, use_container_width=True):
            d = blank(); d.update({"step": "coach_mode", "role": role, "skills": skills, "roadmap_data": rd, "topics": rd["topics"],
                                   "exam_duration": rd["exam_duration_minutes"], "interview_scores": [6.0, 7.5, 8.0],
                                   "quiz_scores": [40, 60, 80], "mock_scores": [55, 62],
                                   "topic_stats": {rd["topics"][0]: [2, 5], rd["topics"][1]: [4, 5]}})
            st.session_state.user_data = d; reset_transient(); save()
            st.session_state.messages = [{"role": "assistant", "content": f"Welcome! I loaded the **{role}** demo. Check 'My Progress'!", "is_new": True}]
            st.rerun()
    st.divider()
    st.markdown("### 🎯 Readiness")
    if U().get("roadmap_data"):
        r = readiness(); st.metric("Score", f"{r['total']}/100"); st.progress(r["total"] / 100)
        st.caption(f"🔥 {streak()}-day streak")
    else: st.info("Finish setup to see your score.")
    st.divider()
    mode = st.radio("🎮 Mode", ["💬 Coach", "🎤 Voice Interview", "📝 Mock Paper", "🧠 Rapid Quiz",
                                "❓ Doubt Solver", "📚 My Library", "🛠️ Skill Builder", "🗓️ Study Plan", "📈 My Progress"], key="mode")
    st.selectbox("Madam Shama's accent", ["co.uk (British)", "com (American)", "com.au (Australian)"], key="accent")
    st.selectbox("🎤 Speaking language", ["en-IN (English, India)", "en-US (English, US)", "en-GB (English, UK)", "hi-IN (Hindi)", "te-IN (Telugu)"], key="lang")
    st.checkbox("🔊 Read replies aloud", value=True, key="tts_on")
    st.checkbox("🌐 Coach uses Google Search (latest info)", key="use_search")

# ==========================================
# 7. MODES
# ==========================================
def show_chat():
    for m in st.session_state.messages:
        with st.chat_message(m["role"], avatar="👩‍🏫" if m["role"] == "assistant" else "🧑"):
            st.markdown(m["content"])
            if m["role"] == "assistant" and m.get("is_new"):
                speak(m["content"]); m["is_new"] = False

SCHEMA = ('{"summary":"warm text","exam_duration_minutes":120,"topics":["t1","t2","t3","t4","t5"],'
          '"gaps":[{"skill":"","why":"","priority":"High"}],"roadmap":[{"week":1,"focus":"","tasks":["",""]}]}')

def handle_coach(text):
    st.session_state.messages.append({"role": "user", "content": text})
    u = U(); step = u["step"]
    with st.spinner("Madam Shama is thinking..."):
        if step == "ask_role":
            u["role"] = text; u["step"] = "ask_skills"; save()
            reply = ask(f"The student wants to prepare for: '{text}'. Acknowledge warmly in 2 sentences, then ask about their current background and skills.")
        elif step == "ask_skills":
            u["skills"] = text
            d = jload(ask(f"Goal: {u['role']}. Background: {text}. Create a 4-week roadmap, 5 core topics, and the standard exam duration in minutes "
                          f"for this role. Return ONLY JSON in this shape: {SCHEMA}", json_mode=True))
            if isinstance(d, dict) and d.get("roadmap"):
                u["roadmap_data"] = d; u["topics"] = d.get("topics", [])
                try: u["exam_duration"] = int(d.get("exam_duration_minutes", 60))
                except Exception: u["exam_duration"] = 60
                u["step"] = "coach_mode"; save()
                reply = f"{d.get('summary', '')}\n\nYour **roadmap** is ready (see 🗓️ Study Plan). Shall we start with a quiz?"
            else: reply = "Sorry, I couldn't build your roadmap. Could you tell me your skills again?"
        else:
            reply = coach_reply(text)
    st.session_state.messages.append({"role": "assistant", "content": reply, "is_new": True})
    touch(); save(); st.rerun()

def mode_coach():
    if U()["step"] == "ask_role":
        st.write("### How Madam Shama Works")
        c = st.columns(3)
        for col, (i, t, d) in zip(c, [("🎯", "1. Set Your Goal", "Tell me your dream job or exam."),
                                       ("🗺️", "2. Get Your Roadmap", "A personalised 4-week plan."),
                                       ("🎙️", "3. Practice Live", "Interviews, mock papers, quizzes, doubts.")]):
            col.markdown(f'<div class="feature-card"><div style="font-size:36px">{i}</div><b>{t}</b><p>{d}</p></div>', unsafe_allow_html=True)
        st.divider()
    show_chat()
    live_mic()
    fb = audio_fallback("coach")
    text = st.chat_input("Type, or tap 🎤 and speak…") or fb
    if text: handle_coach(text)

QSCHEMA = '[{"topic":"","question":"","options":["A) ..","B) ..","C) ..","D) .."],"answer":"A","explanation":""}]'

def valid_mcq(q): return isinstance(q, dict) and all(k in q for k in ("question", "options", "answer")) and len(q["options"]) >= 2

def mode_quiz():
    st.subheader("🧠 Rapid Quiz")
    if need_setup(): return
    q = st.session_state.get("quiz")
    if not q:
        topics = ["Mixed"] + U()["topics"]
        w = weak_topics()
        if w: st.caption(f"💡 Madam Shama suggests practising: **{w[0]}**")
        topic = st.selectbox("Topic", topics, index=topics.index(w[0]) if w and w[0] in topics else 0)
        if st.button("🚀 Start 5-question quiz", type="primary"):
            with st.spinner("Preparing questions..."):
                qs = jload(ask(f"Generate 5 MCQs for a student preparing for {U()['role']}. Topic: {topic if topic != 'Mixed' else ', '.join(U()['topics'])}. "
                               f"Set the topic field to the specific topic. Return ONLY a JSON array like: {QSCHEMA}", json_mode=True))
            qs = [x for x in (qs or []) if valid_mcq(x)]
            if qs: st.session_state["quiz"] = {"qs": qs, "id": time.time(), "done": False}; st.rerun()
            else: st.error("Couldn't generate the quiz. Try again.")
        return
    if not q["done"]:
        with st.form("quizform"):
            picks = [st.radio(f"**Q{i+1}. {x['question']}**", x["options"], index=None, key=f"qz{q['id']}_{i}") for i, x in enumerate(q["qs"])]
            if st.form_submit_button("Submit quiz", type="primary"):
                res = []
                for x, p in zip(q["qs"], picks):
                    ok = bool(p) and p.strip().upper()[:1] == str(x["answer"]).strip().upper()[:1]
                    rec_topic(x.get("topic", "General"), ok); res.append((x, p, ok))
                pct = round(100 * sum(r[2] for r in res) / len(res))
                U()["quiz_scores"].append(pct); touch(); save()
                q.update(done=True, res=res, pct=pct); st.rerun()
    else:
        st.success(f"🎉 You scored {q['pct']}%")
        if q["pct"] >= 80: st.balloons()
        for x, p, ok in q["res"]:
            with st.expander(("✅ " if ok else "❌ ") + x["question"]):
                st.write(f"Your answer: {p or 'not answered'}"); st.write(f"Correct: **{x['answer']}**"); st.write(x.get("explanation", ""))
        if st.button("🔁 New quiz"): st.session_state.pop("quiz"); st.rerun()

PSCHEMA = ('[{"type":"mcq","topic":"","question":"","options":["A) ..","B) ..","C) ..","D) .."],"answer":"A","marks":1},'
           '{"type":"short","topic":"","question":"","model_answer":"","marks":4}]')

def grade_mock(m, answers):
    res, shorts = [], []
    for i, (q, a) in enumerate(zip(m["qs"], answers)):
        mx = float(q.get("marks", 1)); topic = q.get("topic", "General")
        if q.get("type") == "mcq":
            ok = bool(a) and a.strip().upper()[:1] == str(q.get("answer", "")).strip().upper()[:1]
            res.append({"q": q, "a": a, "got": mx if ok else 0.0, "max": mx, "fb": "" if ok else f"Correct answer: {q.get('answer')}", "topic": topic})
        else:
            res.append({"q": q, "a": a, "got": 0.0, "max": mx, "fb": "" if (a and a.strip()) else "No answer given.", "topic": topic})
            if a and a.strip(): shorts.append({"id": i, "question": q["question"], "model_answer": q.get("model_answer", ""), "student_answer": a, "max_marks": mx})
    if shorts:
        g = jload(ask('Grade these short answers fairly against the model answers; partial credit allowed. Return ONLY a JSON array: '
                      '[{"id":0,"awarded":0,"feedback":"one or two sentences"}]\n' + json.dumps(shorts), json_mode=True))
        for x in (g if isinstance(g, list) else []):
            try:
                r = res[int(x["id"])]; r["got"] = min(r["max"], max(0.0, float(x["awarded"]))); r["fb"] = x.get("feedback", "")
            except Exception: pass
    return res

def mode_mock():
    st.subheader("📝 Full Mock Paper (timed)")
    if need_setup(): return
    m = st.session_state.get("mock")
    if not m:
        st.info(f"👩‍🏫 For **{U()['role']}** I suggest **{U()['exam_duration']} minutes**. You can adjust it.")
        n = st.select_slider("Number of questions", [5, 10, 15, 20], value=10)
        mins = st.number_input("Minutes", 5, 240, min(max(U()["exam_duration"], 5), 240))
        if st.button("🚀 Generate paper & start timer", type="primary"):
            with st.spinner("Setting your paper..."):
                qs = jload(ask(f"Create an exam paper of {n} questions for a student preparing for {U()['role']}, covering: {', '.join(U()['topics'])}. "
                               f"About 70% MCQ (1 mark) and 30% short-answer (4 marks). Return ONLY a JSON array like: {PSCHEMA}", json_mode=True))
            qs = [x for x in (qs or []) if isinstance(x, dict) and x.get("question")]
            if qs: st.session_state["mock"] = {"qs": qs, "end": time.time() + mins * 60, "id": time.time(), "done": False}; st.rerun()
            else: st.error("Couldn't generate the paper. Try again.")
        return
    if not m["done"]:
        timer(max(0, m["end"] - time.time()))
        with st.form("mockform"):
            answers = []
            for i, q in enumerate(m["qs"]):
                lab = f"**Q{i+1}. ({q.get('marks', 1)} marks)** {q['question']}"
                k = f"mq{m['id']}_{i}"
                if q.get("type") == "mcq": answers.append(st.radio(lab, q.get("options", []), index=None, key=k))
                else: answers.append(st.text_area(lab, key=k)); live_mic("focus")
            if st.form_submit_button("✅ Submit paper", type="primary"):
                with st.spinner("Madam Shama is marking your paper..."):
                    res = grade_mock(m, answers)
                    got, mx = sum(r["got"] for r in res), sum(r["max"] for r in res)
                    pct = round(100 * got / mx) if mx else 0
                    for r in res: rec_topic(r["topic"], r["got"] >= 0.6 * r["max"])
                    summary = "; ".join(f"{r['topic']}: {r['got']:.0f}/{r['max']:.0f}" for r in res)
                    analysis = ask(f"The student scored {pct}% on a mock paper for {U()['role']}. Per-question results: {summary}. "
                                   "Give 3 short bullets: what went well, what to fix, and a 1-day action plan.")
                    U()["mock_scores"].append(pct); touch(); save()
                m.update(done=True, res=res, pct=pct, got=got, mx=mx, analysis=analysis, late=time.time() > m["end"]); st.rerun()
    else:
        st.metric("Your score", f"{m['got']:.0f}/{m['mx']:.0f}  ({m['pct']}%)")
        if m["late"]: st.warning("⚠️ Submitted after the timer ended.")
        st.markdown(m["analysis"])
        for i, r in enumerate(m["res"]):
            with st.expander(f"Q{i+1}: {r['got']:.0f}/{r['max']:.0f} — {r['q']['question'][:70]}"):
                st.write(f"Your answer: {r['a'] or 'not answered'}")
                if r["fb"]: st.write(r["fb"])
        if st.button("🔁 New paper"): st.session_state.pop("mock"); st.rerun()

def read_upload(f):
    if f.type == "application/pdf":
        reader = PyPDF2.PdfReader(io.BytesIO(f.getvalue()))
        return "\n".join((p.extract_text() or "") for p in reader.pages)[:8000], None
    return "", (f.getvalue(), f.type)

def mode_doubt():
    st.subheader("❓ Doubt Solver")
    st.write("Stuck on a question? Type it or upload a photo/PDF.")
    q = st.text_area("Your question or doubt:")
    live_mic("focus")
    f = st.file_uploader("Optional: image or PDF", type=["jpg", "jpeg", "png", "pdf"])
    level = st.radio("How much help?", ["💡 Hint only", "🪜 Step-by-step", "✅ Full solution"], horizontal=True)
    if st.button("Ask Madam Shama", type="primary") and (q.strip() or f):
        text, media = ("", None)
        if f:
            try: text, media = read_upload(f)
            except Exception as e: st.error(f"Couldn't read the file: {e}")
        system = (PERSONA + " For doubts: 'Hint only' = one nudge plus a guiding question; 'Step-by-step' = explain the approach in steps and "
                  "let the student attempt the last step; 'Full solution' = a full worked solution with the key idea highlighted and one similar practice question.")
        with st.spinner("Thinking..."):
            st.session_state["doubt"] = ask(f"Help level: {level}. Student's goal: {U()['role'] or 'general'}.\nQuestion: {q}\n{text}", system=system, media=media)
        touch(); save()
    if st.session_state.get("doubt"):
        with st.chat_message("assistant", avatar="👩‍🏫"): st.markdown(st.session_state["doubt"])

IVSCHEMA = '{"score":7,"feedback":"2-3 kind but honest sentences","tip":"one concrete improvement, e.g. use the STAR method","next_question":""}'

def mode_interview():
    st.markdown('<div class="meet-room"><div class="video-grid"><div class="video-card active"><div class="avatar">🧑</div><div class="name-tag">You (Candidate)</div></div>'
                '<div class="video-card"><div class="avatar">👩‍🏫</div><div class="name-tag">Madam Shama (Interviewer)</div></div></div></div>', unsafe_allow_html=True)
    if need_setup(): return
    iv = st.session_state.get("iv")
    if not iv:
        kind = st.radio("Interview type", ["HR / Behavioural", "Technical / Subject", "Mixed"], horizontal=True)
        if st.button("▶️ Start interview", type="primary"):
            with st.spinner("Madam Shama is joining..."):
                first = ask(f"Ask the first {kind} interview question for a {U()['role']} role. Return only the question.")
            st.session_state["iv"] = {"kind": kind, "q": first, "log": [], "speak": True, "report": None}; st.rerun()
        return
    if iv["report"]:
        st.markdown(iv["report"])
        if st.button("🔁 New interview"): st.session_state.pop("iv"); st.rerun()
        return
    if iv["log"]:
        s = [x["score"] for x in iv["log"]]
        st.info(f"📊 Answered: {len(s)} | Average: {sum(s)/len(s):.1f}/10")
        last = iv["log"][-1]; st.success(f"**Score {last['score']}/10** — {last['feedback']}\n\n💡 {last['tip']}")
    with st.chat_message("assistant", avatar="👩‍🏫"):
        st.markdown(iv["q"])
        if iv["speak"]: speak(iv["q"]); iv["speak"] = False
    live_mic()
    fb = audio_fallback("iv")
    ans = st.chat_input("Your answer — type or tap 🎤…") or fb
    if st.button("🏁 End interview & get report"):
        if iv["log"]:
            with st.spinner("Writing your report..."):
                iv["report"] = ask("Write a short interview report for the candidate from this log: " + json.dumps(iv["log"]) +
                                   ". Give: overall impression, 3 strengths, 3 things to improve, and 3 practice tips.")
        else: iv["report"] = "No answers yet. Start again when you are ready!"
        st.rerun()
    if ans:
        who = "HR Manager" if iv["kind"].startswith("HR") or (iv["kind"] == "Mixed" and len(iv["log"]) % 2 == 0) else "Senior Interviewer"
        with st.spinner("Evaluating your answer..."):
            d = jload(ask(f"You are a {who} interviewing for {U()['role']}. Question: {iv['q']}\nCandidate answer: {ans}\n"
                          f"Score 0-10 honestly. Return ONLY JSON like: {IVSCHEMA}", json_mode=True))
        if isinstance(d, dict) and "score" in d:
            try: sc = float(d["score"])
            except Exception: sc = 0.0
            iv["log"].append({"q": iv["q"], "a": ans, "score": sc, "feedback": d.get("feedback", ""), "tip": d.get("tip", "")})
            U()["interview_scores"].append(sc); touch(); save()
            iv["q"] = d.get("next_question") or "Tell me about a challenge you overcame."; iv["speak"] = True
        else: st.error("Couldn't evaluate that answer. Please try again.")
        st.rerun()

PROJSCHEMA = '{"title":"","why":"","steps":["","",""],"deliverable":"what to show as proof","search_keywords":["",""]}'

def mode_skills():
    st.subheader("🛠️ Practical Skill Builder")
    if need_setup(): return
    gaps = (U().get("roadmap_data") or {}).get("gaps", [])
    if gaps:
        st.write("**🎯 Based on your roadmap gaps:**")
        cols = st.columns(len(gaps))
        for i, g in enumerate(gaps):
            if cols[i].button(f"Learn: {g.get('skill', '?')}", use_container_width=True, key=f"gap{i}"): st.session_state["skill_pick"] = g.get("skill")
    custom = st.text_input("Or type a skill to practise:")
    if custom and st.button("Create project"): st.session_state["skill_pick"] = custom
    skill = st.session_state.pop("skill_pick", None)
    if skill:
        with st.spinner(f"Designing a project for {skill}..."):
            d = jload(ask(f"Create a practical 3-step beginner mini-project to learn '{skill}' for someone preparing for {U()['role']}. "
                          f"Do NOT invent URLs; give search keywords instead. Return ONLY JSON like: {PROJSCHEMA}", json_mode=True))
        if isinstance(d, dict) and d.get("steps"): st.session_state["proj"] = {"skill": skill, "d": d, "id": time.time(), "fb": None}
        else: st.error("Couldn't design a project. Try again.")
    p = st.session_state.get("proj")
    if not p: return
    d = p["d"]; st.write("---"); st.markdown(f"### 📋 {d.get('title', p['skill'])}"); st.write(d.get("why", ""))
    ticks = [st.checkbox(s, key=f"ps{p['id']}_{i}") for i, s in enumerate(d["steps"])]
    st.write(f"**Deliverable:** {d.get('deliverable', '')}")
    if d.get("search_keywords"): st.caption("🔎 Search Google for: " + " · ".join(d["search_keywords"]))
    work = st.text_area("Describe or paste what you built / learned:", key=f"pw{p['id']}")
    live_mic("focus")
    if st.button("📤 Submit for Madam Shama's review", type="primary", disabled=not all(ticks)):
        r = jload(ask(f"Project: {d.get('title')}. Steps: {d['steps']}. Deliverable expected: {d.get('deliverable')}. Student's submission: {work}\n"
                      'Review kindly. Return ONLY JSON: {"passed":true,"feedback":"2-3 sentences"}', json_mode=True))
        if isinstance(r, dict):
            p["fb"] = r.get("feedback", "")
            if r.get("passed") and not p.get("awarded"):
                U()["skill_projects_completed"] += 1; p["awarded"] = True; touch(); save()
                st.success("Fantastic! +2 Readiness points.")
        st.rerun()
    if p.get("fb"): st.info(p["fb"])

def esc(s): return str(s).replace("\\", "\\\\").replace(",", "\\,").replace(";", "\\;").replace("\n", "\\n")

def make_ics(rd):
    start = date.today() + timedelta(days=1)
    L = ["BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//Madam Shama//EN"]
    for w in rd.get("roadmap", []):
        try: n = int(w.get("week", 1))
        except Exception: continue
        d0 = start + timedelta(days=7 * (n - 1)); d1 = d0 + timedelta(days=7)
        L += ["BEGIN:VEVENT", f"UID:shama-{st.session_state.uid}-w{n}-{d0:%Y%m%d}@madamshama",
              f"DTSTAMP:{datetime.utcnow():%Y%m%dT%H%M%SZ}", f"DTSTART;VALUE=DATE:{d0:%Y%m%d}", f"DTEND;VALUE=DATE:{d1:%Y%m%d}",
              f"SUMMARY:{esc('Week %d: %s' % (n, w.get('focus', '')))}", f"DESCRIPTION:{esc(chr(10).join(w.get('tasks', [])))}", "END:VEVENT"]
    L.append("END:VCALENDAR"); return "\r\n".join(L)

def mode_plan():
    st.subheader("🗓️ Your Study Plan")
    if need_setup(): return
    rd = U().get("roadmap_data") or {}
    if rd.get("summary"): st.info(rd["summary"])
    for g in rd.get("gaps", []): st.write(f"🎯 **{g.get('skill')}** ({g.get('priority')}) — {g.get('why')}")
    total = done = 0
    for w in rd.get("roadmap", []):
        with st.expander(f"Week {w.get('week')}: {w.get('focus', '')}", expanded=True):
            for i, t in enumerate(w.get("tasks", [])):
                k = f"w{w.get('week')}-{i}"; total += 1
                v = st.checkbox(t, value=U()["done"].get(k, False), key=f"chk_{k}_{hash(t)}")
                if v != U()["done"].get(k, False): U()["done"][k] = v; touch(); save()
                done += 1 if v else 0
    if total: st.progress(done / total); st.caption(f"{done}/{total} tasks done")
    if rd.get("roadmap"):
        st.download_button("📅 Download plan for Google Calendar (.ics)", make_ics(rd), "study_plan.ics", "text/calendar")

def mode_progress():
    st.subheader("📈 Your Growth Analytics")
    c1, c2, c3 = st.columns(3)
    for col, title, key, color in [(c1, "🎤 Interviews (/10)", "interview_scores", "#8b5cf6"), (c2, "🧠 Quizzes (%)", "quiz_scores", "#10b981"),
                                   (c3, "📝 Mock papers (%)", "mock_scores", "#f59e0b")]:
        with col:
            st.markdown(f"**{title}**"); s = U()[key]
            if s: st.line_chart(pd.DataFrame({"Attempt": range(1, len(s) + 1), "Score": s}).set_index("Attempt"), color=color)
            else: st.info("No data yet.")
    st.markdown("### 🧩 Topic accuracy")
    ts = U()["topic_stats"]
    if ts: st.bar_chart(pd.DataFrame({"Topic": list(ts), "Accuracy %": [round(100 * c / n) for c, n in ts.values()]}).set_index("Topic"))
    else: st.info("Take a quiz or mock paper to see your strong and weak topics.")
    if U().get("roadmap_data"):
        r = readiness(); st.markdown(f"### 🏆 Readiness: **{r['total']}/100**  |  🔥 {streak()}-day streak")
        for k, (v, mx) in r["parts"].items(): st.write(f"- {k}: {v}/{mx}")

def read_material(blob):
    data = blob.download_as_bytes(); n = blob.name.lower()
    if n.endswith(".txt"): return data.decode("utf-8", "ignore")[:12000], None
    if n.endswith(".pdf"):
        try: text = "\n".join((p.extract_text() or "") for p in PyPDF2.PdfReader(io.BytesIO(data)).pages)[:12000]
        except Exception: text = ""
        return (text, None) if len(text.strip()) > 50 else ("", (data, "application/pdf"))
    return "", (data, "image/png" if n.endswith(".png") else "image/jpeg")

def mode_library():
    st.subheader("📚 My Library")
    st.write("Upload your notes, syllabus or past papers. Madam Shama can summarise them, answer questions and quiz you from them.")
    if not BUCKET:
        st.info("Cloud Storage isn't connected yet. Follow README.md to turn this on."); return
    prefix = f"users/{st.session_state.uid}/"
    ups = st.file_uploader("Upload files (PDF, image, text; max 10 MB each)", type=["pdf", "png", "jpg", "jpeg", "txt"], accept_multiple_files=True)
    if ups and st.button("⬆️ Save to my library", type="primary"):
        for f in ups:
            if f.size > 10 * 1024 * 1024: st.warning(f"{f.name} is over 10 MB, skipped."); continue
            BUCKET.blob(prefix + re.sub(r"[^A-Za-z0-9._-]", "_", f.name)).upload_from_string(f.getvalue(), content_type=f.type)
        touch(); save(); st.success("Saved to your library!")
    blobs = list(BUCKET.list_blobs(prefix=prefix))
    if not blobs: st.info("Your library is empty."); return
    b = st.selectbox("Your files", blobs, format_func=lambda x: x.name.split("/")[-1])
    act = st.radio("What should Madam Shama do?", ["📝 Summarise", "🧠 Quiz me from this", "❓ Ask a question"], horizontal=True)
    q = ""
    if act.startswith("❓"):
        q = st.text_input("Your question:"); live_mic("focus")
    c1, c2 = st.columns(2)
    if c1.button("▶️ Go", type="primary", use_container_width=True):
        with st.spinner("Reading your file..."):
            text, media = read_material(b)
            if act.startswith("📝"):
                st.session_state["lib_out"] = ask(f"Summarise this study material in 6 short bullets, then list 5 key terms.\n\n{text}", media=media)
            elif act.startswith("❓"):
                st.session_state["lib_out"] = ask(f"Using ONLY this study material, answer the student's question. If it isn't covered, say so.\nQuestion: {q}\n\n{text}", media=media)
            else:
                qs = [x for x in (jload(ask(f"Generate 5 MCQs strictly from this study material. Set topic to the specific topic. Return ONLY a JSON array like: {QSCHEMA}\n\nMaterial:\n{text}",
                                            json_mode=True, media=media)) or []) if valid_mcq(x)]
                if qs:
                    st.session_state["quiz"] = {"qs": qs, "id": time.time(), "done": False}
                    st.session_state["lib_out"] = "✅ Your quiz is ready! Open **🧠 Rapid Quiz** in the sidebar to take it."
                else: st.session_state["lib_out"] = "Sorry, I couldn't make a quiz from that file. Try another."
        touch(); save()
    if c2.button("🗑️ Delete this file", use_container_width=True):
        b.delete(); st.session_state.pop("lib_out", None); st.rerun()
    if st.session_state.get("lib_out"):
        with st.chat_message("assistant", avatar="👩‍🏫"): st.markdown(st.session_state["lib_out"])

# ==========================================
# 8. ROUTER
# ==========================================
{"💬 Coach": mode_coach, "🎤 Voice Interview": mode_interview, "📝 Mock Paper": mode_mock, "🧠 Rapid Quiz": mode_quiz,
 "❓ Doubt Solver": mode_doubt, "📚 My Library": mode_library, "🛠️ Skill Builder": mode_skills, "🗓️ Study Plan": mode_plan, "📈 My Progress": mode_progress}[mode]()
