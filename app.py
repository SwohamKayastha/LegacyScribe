"""
LegacyScribe — Family Memory Agent
Gradio Space entry point
"""

import gradio as gr
import json
import os
from pathlib import Path
from huggingface_hub import hf_hub_download, login
from llama_cpp import Llama

# ── Auth ──────────────────────────────────────────────────────────────────────
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    login(token=hf_token)

# ── Model loading ─────────────────────────────────────────────────────────────
MODEL_REPO   = os.environ.get("MODEL_REPO", "build-small-hackathon/legacystribe-Qwen3.5-9B.Q4_K_M")
MODEL_FILE   = os.environ.get("MODEL_FILE", "Qwen3.5-9B.Q4_K_M.gguf")
N_CTX        = 4096
N_GPU_LAYERS = int(os.environ.get("N_GPU_LAYERS", "0"))

print(f"Loading model from {MODEL_REPO}/{MODEL_FILE}...")
model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)
llm = Llama(model_path=model_path, n_ctx=N_CTX, n_gpu_layers=N_GPU_LAYERS, verbose=False)
print("Model ready.")

# ── System prompts ────────────────────────────────────────────────────────────
SYSTEM_PROMPTS = {
    "questioner": (
        "You are a gentle memory guide helping an elderly person tell their life story. "
        "Ask exactly one warm, open follow-up question. Never ask more than one question. "
        "Be patient, kind, and culturally sensitive to Nepali and South Asian contexts."
    ),
    "extractor": (
        "You are an extractor agent. Given a memory fragment, extract structured information "
        "as JSON with keys relevant to the content (who, when, where, what, emotion). "
        "Output only valid JSON, nothing else."
    ),
    "arcdetector": (
        "You are an arc detector agent. Given a memory fragment, identify the narrative stage. "
        "Output one word only: setup, tension, turn, or meaning."
    ),
    "publisher": (
        "You are a publisher agent. Given memory notes, synthesize them into a single warm, "
        "narrative paragraph suitable for a family memory book. Write in first person. "
        "Use natural, unhurried language. Output only the paragraph, nothing else."
    ),
}

def call_agent(agent: str, user_text: str, max_tokens: int = 512, temp: float = 0.4) -> str:
    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPTS[agent]},
            {"role": "user",   "content": user_text},
        ],
        max_tokens=max_tokens,
        temperature=temp,
    )
    return response["choices"][0]["message"]["content"].strip()

# ── Session state ─────────────────────────────────────────────────────────────
def init_state():
    return {"turns": [], "all_notes": [], "chapter": "", "chapter_num": 1}

def process_turn(user_input: str, state: dict):
    if not user_input.strip():
        return state, "", "", "", []

    arc = call_agent("arcdetector", user_input, max_tokens=10, temp=0.1)
    arc = arc.lower().strip()
    if arc not in ("setup", "tension", "turn", "meaning"):
        arc = "setup"

    try:
        raw = call_agent("extractor", user_input, max_tokens=200, temp=0.1)
        notes = json.loads(raw)
        note_lines = [f"{k}: {v}" for k, v in notes.items() if v]
    except Exception:
        note_lines = [user_input[:120]]

    state["all_notes"].extend(note_lines)

    history_ctx = "\n".join(f"Memory: {t['user']}" for t in state["turns"][-3:])
    question_ctx = f"{history_ctx}\nMemory: {user_input}" if history_ctx else f"Memory: {user_input}"
    question = call_agent("questioner", question_ctx, max_tokens=80, temp=0.7)

    chapter = state["chapter"]
    prev_chapter = chapter
    if len(state["all_notes"]) >= 3:
        notes_text = "\n".join(f"{i+1}. {n}" for i, n in enumerate(state["all_notes"][-6:]))
        chapter = call_agent("publisher", f"Notes:\n{notes_text}", max_tokens=400, temp=0.4)
        if chapter != prev_chapter and prev_chapter:
            state["chapter_num"] += 1
        state["chapter"] = chapter

    state["turns"].append({"user": user_input, "question": question, "arc": arc, "notes": note_lines})
    return state, question, arc, chapter, note_lines

# ── HTML builders ─────────────────────────────────────────────────────────────
ARC_ICONS = {"setup": "◎", "tension": "◈", "turn": "◉", "meaning": "✦"}
ARC_LABELS = {"setup": "Setting the scene", "tension": "Something changed", "turn": "A new direction", "meaning": "What it meant"}

def build_chat_html(turns):
    if not turns:
        return '''
        <div class="ls-chat" id="ls-chat">
          <div class="ls-welcome">
            <div class="ls-welcome-icon">📖</div>
            <div class="ls-welcome-text">Tell me a memory — anything at all.<br>A meal, a person, a festival, a place.<br><em>I am here to listen.</em></div>
          </div>
        </div>'''
    html = '<div class="ls-chat" id="ls-chat">'
    for t in turns:
        arc = t.get("arc", "setup")
        icon = ARC_ICONS.get(arc, "◎")
        label = ARC_LABELS.get(arc, arc)
        html += f'''
        <div class="ls-turn">
          <div class="ls-bubble-user">{t["user"]}</div>
          <div class="ls-arc-pill {arc}"><span>{icon}</span><span>{label}</span></div>
          <div class="ls-bubble-agent">{t["question"]}</div>
        </div>'''
    html += '</div>'
    html += '<script>setTimeout(function(){var c=document.getElementById("ls-chat");if(c)c.scrollTop=c.scrollHeight;},80);</script>'
    return html

def build_notes_html(notes):
    if not notes:
        return ''
    html = '<div class="ls-fragments-label">Memory fragments extracted</div><div class="ls-fragments">'
    for n in notes[-5:]:
        parts = n.split(": ", 1)
        if len(parts) == 2:
            html += f'<div class="ls-fragment"><span class="ls-fkey">{parts[0]}</span><span class="ls-fval">{parts[1]}</span></div>'
        else:
            html += f'<div class="ls-fragment"><span class="ls-fval">{n}</span></div>'
    html += '</div>'
    return html

LINE_H = 28   # px per ruled line

def build_page_html(chapter, chapter_num):
    if not chapter:
        return '''
        <div class="ls-book-wrap">
          <div class="ls-book-cover-hint">— waiting for your story —</div>
          <div class="ls-page" id="ls-page">
            <div class="ls-page-inner">
              <div class="ls-chapter-label">Chapter One</div>
              <div class="ls-ruled-area">
                <div class="ls-ruled-lines"></div>
                <div class="ls-page-empty">Your memories are gathering.<br>Share a few more and I will begin to write.</div>
              </div>
            </div>
            <div class="ls-spine"></div>
            <div class="ls-page-shadow"></div>
          </div>
        </div>'''

    words = chapter.split()
    lines, current = [], []
    chars = 0
    for w in words:
        if chars + len(w) + 1 > 52:
            lines.append(" ".join(current))
            current, chars = [w], len(w)
        else:
            current.append(w)
            chars += len(w) + 1
    if current:
        lines.append(" ".join(current))

    text_html = ""
    for line in lines:
        text_html += f'<div class="ls-text-line">{line}</div>'

    return f'''
    <div class="ls-book-wrap">
      <div class="ls-page" id="ls-page">
        <div class="ls-page-inner">
          <div class="ls-chapter-label">Chapter {chapter_num}</div>
          <div class="ls-ruled-area">
            <div class="ls-ruled-lines"></div>
            <div class="ls-text-block">{text_html}</div>
          </div>
        </div>
        <div class="ls-spine"></div>
        <div class="ls-page-curl"></div>
        <div class="ls-page-shadow"></div>
      </div>
    </div>
    <script>
      (function(){{
        var p = document.getElementById("ls-page");
        if(!p) return;
        p.classList.remove("ls-flip");
        void p.offsetWidth;
        p.classList.add("ls-flip");
      }})();
    </script>'''

# ── CSS / HEAD ─────────────────────────────────────────────────────────────────
HEAD = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,500;0,700;1,400;1,500&family=Source+Serif+4:ital,opsz,wght@0,8..60,300;0,8..60,400;1,8..60,300;1,8..60,400&family=JetBrains+Mono:wght@300;400&display=swap" rel="stylesheet">
<style>
/* ── Reset ── */
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0;}

:root{
  --walnut:   #241208;
  --walnut2:  #2E1810;
  --parchment:#F0E4CC;
  --parchment2:#FAF4E8;
  --ink:      #1C0E06;
  --gold:     #C8861E;
  --gold2:    #E8A830;
  --sage:     #6A9070;
  --dust:     #B8A484;
  --cream:    #F8F2E4;
  --red:      #C04030;
  --purple:   #9060A8;
  --shadow:   rgba(20,8,2,0.5);
  --line-h:   28px;
}

html,body,.gradio-container{
  background: var(--walnut) !important;
  font-family:'Source Serif 4',Georgia,serif !important;
  color:var(--parchment) !important;
  min-height:100vh;
}

/* ── Suppress Gradio chrome ── */
footer,.svelte-1rjryqp,#footer{display:none!important;}
.gradio-container{padding:0!important;max-width:100%!important;}
.gr-prose{color:var(--parchment)!important;}
.gr-button{font-family:'Source Serif 4',serif!important;}

/* ════════════════════════════════
   HEADER
════════════════════════════════ */
.ls-header{
  text-align:center;
  padding:3rem 1rem 2rem;
  background: linear-gradient(to bottom, #1A0A02, var(--walnut));
  border-bottom:1px solid rgba(200,134,30,0.25);
  position:relative;
  overflow:hidden;
}
.ls-header::before{
  content:'';
  position:absolute;
  inset:0;
  background: radial-gradient(ellipse 60% 40% at 50% 0%, rgba(200,134,30,0.08), transparent);
  pointer-events:none;
}
.ls-wordmark{
  font-family:'Playfair Display',serif;
  font-size:clamp(2.4rem,6vw,4rem);
  font-weight:700;
  letter-spacing:-0.03em;
  line-height:1;
  position:relative;
}
.ls-wordmark-legacy{ color:var(--gold2); }
.ls-wordmark-scribe{ color:var(--parchment); font-style:italic; font-weight:500; }
.ls-wordmark-dot{ color:var(--gold); font-style:normal; }

.ls-deco-line{
  width:80px; height:1px;
  background:linear-gradient(to right, transparent, var(--gold), transparent);
  margin:0.9rem auto 0.7rem;
}
.ls-tagline{
  font-size:0.95rem;
  color:var(--dust);
  font-style:italic;
  letter-spacing:0.05em;
}
.ls-badges{
  display:flex;gap:0.5rem;
  justify-content:center;
  margin-top:1.2rem;
  flex-wrap:wrap;
}
.ls-badge{
  font-family:'JetBrains Mono',monospace;
  font-size:0.6rem;
  padding:0.18rem 0.65rem;
  border:1px solid rgba(200,134,30,0.4);
  border-radius:1px;
  color:rgba(200,134,30,0.9);
  letter-spacing:0.1em;
  text-transform:uppercase;
}

/* ════════════════════════════════
   LAYOUT
════════════════════════════════ */
.ls-main{
  display:grid;
  grid-template-columns:1fr 1fr;
  max-width:1280px;
  margin:0 auto;
  min-height:calc(100vh - 220px);
}
@media(max-width:780px){.ls-main{grid-template-columns:1fr;}}

/* ════════════════════════════════
   LEFT — CONVERSATION
════════════════════════════════ */
.ls-conversation{
  border-right:1px solid rgba(200,134,30,0.15);
  padding:2rem 1.75rem;
  display:flex;
  flex-direction:column;
  gap:1.25rem;
  background:linear-gradient(160deg, rgba(46,24,16,0.6), rgba(36,18,8,0.9));
}
.ls-section-label{
  font-family:'JetBrains Mono',monospace;
  font-size:0.6rem;
  letter-spacing:0.18em;
  text-transform:uppercase;
  color:var(--gold);
  opacity:0.8;
}

/* Chat */
.ls-chat{
  display:flex;
  flex-direction:column;
  gap:1.25rem;
  max-height:360px;
  overflow-y:auto;
  padding-right:0.25rem;
  scrollbar-width:thin;
  scrollbar-color:rgba(200,134,30,0.3) transparent;
}
.ls-welcome{
  display:flex;flex-direction:column;align-items:center;
  gap:1rem;padding:2rem 1rem;
  color:var(--dust);text-align:center;
}
.ls-welcome-icon{font-size:2.5rem;opacity:0.6;}
.ls-welcome-text{font-size:0.9rem;line-height:1.7;font-style:italic;}
.ls-welcome-text em{color:var(--gold);font-style:italic;}

.ls-turn{display:flex;flex-direction:column;gap:0.5rem;}

.ls-bubble-user{
  align-self:flex-end;
  background:rgba(200,134,30,0.12);
  border:1px solid rgba(200,134,30,0.25);
  border-radius:14px 14px 3px 14px;
  padding:0.8rem 1.1rem;
  max-width:88%;
  font-size:0.9rem;
  line-height:1.6;
  color:var(--parchment);
  position:relative;
}

.ls-arc-pill{
  align-self:flex-start;
  display:inline-flex;align-items:center;gap:0.35rem;
  font-family:'JetBrains Mono',monospace;
  font-size:0.6rem;
  letter-spacing:0.08em;
  text-transform:uppercase;
  padding:0.18rem 0.7rem;
  border-radius:2px;
  margin-left:0.25rem;
}
.ls-arc-pill.setup   {background:rgba(106,144,112,0.15);color:var(--sage);  border:1px solid rgba(106,144,112,0.3);}
.ls-arc-pill.tension {background:rgba(192,64,48,0.12);  color:#D06858;      border:1px solid rgba(192,64,48,0.25);}
.ls-arc-pill.turn    {background:rgba(200,134,30,0.12); color:var(--gold2); border:1px solid rgba(200,134,30,0.3);}
.ls-arc-pill.meaning {background:rgba(144,96,168,0.12); color:#B080C8;      border:1px solid rgba(144,96,168,0.25);}

.ls-bubble-agent{
  align-self:flex-start;
  background:rgba(240,228,204,0.05);
  border:1px solid rgba(184,164,132,0.15);
  border-radius:3px 14px 14px 14px;
  padding:0.8rem 1.1rem;
  max-width:88%;
  font-size:0.9rem;
  line-height:1.65;
  color:rgba(240,228,204,0.9);
  font-style:italic;
}
.ls-bubble-agent::before{
  content:'✦ ';
  color:var(--gold);
  font-style:normal;
  font-size:0.65rem;
  vertical-align:middle;
}

/* Fragments */
.ls-fragments-label{
  font-family:'JetBrains Mono',monospace;
  font-size:0.58rem;
  letter-spacing:0.15em;
  text-transform:uppercase;
  color:rgba(200,134,30,0.7);
  margin-bottom:0.4rem;
}
.ls-fragments{display:flex;flex-direction:column;gap:0.25rem;}
.ls-fragment{
  display:flex;gap:0.5rem;align-items:baseline;
  font-family:'JetBrains Mono',monospace;
  font-size:0.68rem;
  padding:0.22rem 0.6rem 0.22rem 0.8rem;
  border-left:2px solid rgba(200,134,30,0.5);
  line-height:1.4;
}
.ls-fkey{color:var(--gold);min-width:60px;flex-shrink:0;}
.ls-fval{color:var(--dust);}

/* Input */
.ls-input-wrap textarea{
  width:100%;
  background:rgba(20,8,2,0.7)!important;
  border:1px solid rgba(184,164,132,0.2)!important;
  border-radius:4px!important;
  color:var(--parchment)!important;
  font-family:'Source Serif 4',serif!important;
  font-size:0.92rem!important;
  line-height:1.65!important;
  padding:0.9rem 1.1rem!important;
  resize:none!important;
  transition:border-color 0.2s,box-shadow 0.2s;
}
.ls-input-wrap textarea:focus{
  border-color:rgba(200,134,30,0.5)!important;
  outline:none!important;
  box-shadow:0 0 0 3px rgba(200,134,30,0.08)!important;
}
.ls-input-wrap textarea::placeholder{color:var(--dust)!important;opacity:0.45;}

/* Buttons */
.ls-btn-row{display:flex;gap:0.75rem;align-items:center;}
.ls-btn{
  font-family:'Source Serif 4',serif!important;
  font-size:0.88rem!important;
  font-weight:400!important;
  letter-spacing:0.02em!important;
  padding:0.6rem 1.5rem!important;
  border-radius:3px!important;
  cursor:pointer!important;
  transition:all 0.16s!important;
  border:none!important;
}
.ls-btn-primary{background:var(--gold)!important;color:var(--walnut)!important;font-weight:600!important;}
.ls-btn-primary:hover{background:var(--gold2)!important;transform:translateY(-1px);box-shadow:0 4px 16px rgba(200,134,30,0.25)!important;}
.ls-btn-secondary{background:transparent!important;color:var(--dust)!important;border:1px solid rgba(184,164,132,0.25)!important;}
.ls-btn-secondary:hover{border-color:var(--dust)!important;color:var(--parchment)!important;}

/* Thinking dots */
.ls-thinking{
  display:flex;align-items:center;gap:0.5rem;
  font-size:0.78rem;color:var(--dust);font-style:italic;
  min-height:1.2rem;
}
.ls-dot{
  width:4px;height:4px;border-radius:50%;
  background:var(--gold);
  animation:ls-pulse 1.3s ease-in-out infinite;
}
.ls-dot:nth-child(2){animation-delay:0.2s;}
.ls-dot:nth-child(3){animation-delay:0.4s;}
@keyframes ls-pulse{
  0%,80%,100%{opacity:0.15;transform:scale(0.7);}
  40%{opacity:1;transform:scale(1);}
}

/* ════════════════════════════════
   RIGHT — BOOK
════════════════════════════════ */
.ls-book{
  padding:2.5rem 2rem;
  display:flex;flex-direction:column;gap:1.5rem;
  background:linear-gradient(160deg, rgba(30,14,6,0.95), rgba(24,10,2,0.98));
}

/* Book wrapper — perspective for 3D flip */
.ls-book-wrap{
  perspective:1200px;
  perspective-origin:50% 40%;
}
.ls-book-cover-hint{
  font-family:'JetBrains Mono',monospace;
  font-size:0.6rem;
  letter-spacing:0.15em;
  text-transform:uppercase;
  color:rgba(200,134,30,0.3);
  text-align:center;
  margin-bottom:0.75rem;
}

/* ── The page itself ── */
.ls-page{
  position:relative;
  background: linear-gradient(105deg, #F8F2E0 0%, #F2EAD4 40%, #EDE4C8 100%);
  border-radius:2px 6px 6px 2px;
  min-height:340px;
  box-shadow:
    -6px 0 18px rgba(20,8,2,0.5),
    2px 2px 6px rgba(20,8,2,0.25),
    inset 0 0 60px rgba(180,140,80,0.06);
  transform-origin: left center;
  transform-style: preserve-3d;
  transition: none;
  overflow:hidden;
}

/* Page-flip animation */
@keyframes pageFlip{
  0%  { transform: rotateY(-25deg) skewY(1deg); opacity:0.7; }
  35% { transform: rotateY(-8deg)  skewY(0.3deg); opacity:0.9; }
  65% { transform: rotateY(-3deg)  skewY(0deg); opacity:0.97; }
  100%{ transform: rotateY(0deg)   skewY(0deg); opacity:1; }
}
.ls-page.ls-flip{
  animation: pageFlip 0.55s cubic-bezier(0.23,1,0.32,1) forwards;
}

/* Leather spine */
.ls-spine{
  position:absolute;
  left:0;top:0;bottom:0;width:18px;
  background:linear-gradient(to right,
    #6B4012 0%, #9B6422 30%, #B88030 50%, #9B6422 70%, #6B4012 100%);
  box-shadow:inset -2px 0 4px rgba(0,0,0,0.3), 2px 0 6px rgba(0,0,0,0.2);
}
/* Spine stitching */
.ls-spine::before{
  content:'';
  position:absolute;
  left:7px;top:12px;bottom:12px;width:1px;
  background:repeating-linear-gradient(to bottom,
    rgba(255,220,120,0.5) 0px, rgba(255,220,120,0.5) 4px,
    transparent 4px, transparent 8px);
}

/* Page curl bottom-right */
.ls-page-curl{
  position:absolute;
  bottom:0;right:0;
  width:36px;height:36px;
  background:linear-gradient(135deg,
    transparent 50%,
    rgba(180,140,80,0.15) 50%,
    rgba(140,100,40,0.1) 100%);
  border-top-left-radius:3px;
  pointer-events:none;
}
.ls-page-curl::after{
  content:'';
  position:absolute;
  bottom:0;right:0;
  width:24px;height:24px;
  background:linear-gradient(135deg, transparent 50%, rgba(160,120,60,0.08) 50%);
  border-top-left-radius:2px;
}

/* Drop shadow panel */
.ls-page-shadow{
  position:absolute;
  bottom:-8px;left:12px;right:-4px;height:12px;
  background:rgba(20,8,2,0.3);
  border-radius:0 0 4px 4px;
  filter:blur(6px);
  z-index:-1;
}

/* Page inner content */
.ls-page-inner{
  position:relative;
  padding:1.5rem 1.5rem 1.5rem 2.5rem; /* left pad for spine */
  z-index:1;
}

/* Chapter label */
.ls-chapter-label{
  font-family:'Playfair Display',serif;
  font-size:0.72rem;
  font-weight:700;
  letter-spacing:0.22em;
  text-transform:uppercase;
  color:#8B5E1A;
  margin-bottom:1rem;
  padding-bottom:0.5rem;
  border-bottom:1px solid rgba(180,140,80,0.3);
  position:relative;
}
.ls-chapter-label::after{
  content:'';
  position:absolute;
  bottom:-1px;left:0;width:40px;height:1px;
  background:var(--gold);
}

/* Ruled area — lines + text together */
.ls-ruled-area{
  position:relative;
  min-height:240px;
}
.ls-ruled-lines{
  position:absolute;
  inset:0;
  background-image:repeating-linear-gradient(
    to bottom,
    transparent 0px,
    transparent calc(var(--line-h) - 1px),
    rgba(160,120,60,0.18) calc(var(--line-h) - 1px),
    rgba(160,120,60,0.18) var(--line-h)
  );
  pointer-events:none;
}

/* Text sits ON the ruled lines */
.ls-text-block{
  position:relative;
  z-index:1;
}
.ls-text-line{
  font-family:'Source Serif 4',serif;
  font-size:0.88rem;
  line-height:var(--line-h);
  color:#2A1408;
  font-weight:300;
  height:var(--line-h);
  overflow:hidden;
  white-space:nowrap;
  text-overflow:ellipsis;
  padding-right:0.5rem;
  position:relative;
  top:-1px;
}

.ls-page-empty{
  position:relative;
  z-index:1;
  padding-top:calc(var(--line-h) * 3);
  font-family:'Source Serif 4',serif;
  font-style:italic;
  color:#9A7A50;
  font-size:0.85rem;
  line-height:1.7;
  text-align:center;
  opacity:0.8;
}

/* ── Download button ── */
.ls-dl-btn{
  align-self:flex-start;
  font-family:'JetBrains Mono',monospace!important;
  font-size:0.65rem!important;
  letter-spacing:0.1em!important;
  text-transform:uppercase!important;
  padding:0.4rem 1rem!important;
  background:transparent!important;
  color:rgba(200,134,30,0.7)!important;
  border:1px solid rgba(200,134,30,0.25)!important;
  border-radius:2px!important;
  cursor:pointer!important;
  transition:all 0.16s!important;
}
.ls-dl-btn:hover{
  background:rgba(200,134,30,0.08)!important;
  color:var(--gold2)!important;
  border-color:rgba(200,134,30,0.5)!important;
}

/* ── Footer ── */
.ls-footer{
  text-align:center;
  padding:1.25rem 1rem;
  border-top:1px solid rgba(200,134,30,0.1);
  font-family:'JetBrains Mono',monospace;
  font-size:0.58rem;
  color:rgba(184,164,132,0.4);
  letter-spacing:0.1em;
  text-transform:uppercase;
}
</style>
"""

INTRO_JS = """
() => {
  /* Pure animation — no interaction. Just remove overlay from DOM after CSS is done. */
  /* Total: 3.5s delay + 0.9s fade = 4.4s. Add small buffer. */
  setTimeout(function() {
    var el = document.getElementById('ls-intro');
    if (el) el.style.display = 'none';
  }, 4600);
}
"""

# ── Gradio app ─────────────────────────────────────────────────────────────────
with gr.Blocks(head=HEAD, theme=gr.themes.Base(), css="body{background:#241208!important;}.gradio-container{background:#241208!important;}", js=INTRO_JS) as demo:

    state = gr.State(init_state())

    # ── Book intro overlay — pure CSS animation, no interaction ─────────────────
    gr.HTML("""
    <style>
    /* ── Overlay ── */
    #ls-intro {
      position:fixed; inset:0; z-index:9999; overflow:hidden;
      display:flex; flex-direction:column;
      align-items:center; justify-content:center; gap:1.4rem;
      background:radial-gradient(ellipse 70% 80% at 50% 50%, #2A1408, #160700 70%, #0D0500);
      pointer-events:none;
      /* Step 4 — fade the whole overlay out */
      animation: _ov-out 0.9s ease 3.5s forwards;
    }
    /* Step 4 — golden page-light flash */
    #ls-intro::after {
      content:''; position:absolute; inset:0; z-index:10; pointer-events:none;
      background:radial-gradient(ellipse 55% 50% at 50% 48%,
        rgba(235,175,55,0.55) 0%, rgba(200,134,30,0.2) 42%, transparent 68%);
      opacity:0;
      animation: _flash 0.6s ease 3.5s forwards;
    }

    @keyframes _ov-out { 0%{opacity:1} 20%{opacity:1} 100%{opacity:0} }
    @keyframes _flash  { 0%{opacity:0} 28%{opacity:1} 100%{opacity:0} }

    /* ── Ambient glow — blooms as cover opens ── */
    #ls-i-glow {
      position:absolute; width:520px; height:520px; border-radius:50%; pointer-events:none;
      background:radial-gradient(circle, rgba(200,134,30,0.13) 0%, rgba(200,134,30,0.04) 45%, transparent 70%);
      top:50%; left:50%;
      transform:translate(-50%,-50%) scale(0.3); opacity:0;
      animation:_glow-in 1s ease 1.9s forwards;
    }
    @keyframes _glow-in { to{opacity:1; transform:translate(-50%,-50%) scale(1);} }

    /* ── 3-D scene — Step 4 surges toward viewer ── */
    #ls-i-scene {
      perspective:1400px; perspective-origin:50% 45%;
      /* Step 4 zoom */
      animation: _surge 0.82s cubic-bezier(0.3,0,0.6,1) 3.5s forwards;
    }
    @keyframes _surge { 0%{transform:scale(1)} 100%{transform:scale(8)} }

    /* ── Book container — Step 1 drop-in ── */
    #ls-i-book {
      position:relative; width:300px; height:420px;
      transform-style:preserve-3d;
      filter:drop-shadow(0 28px 56px rgba(10,4,0,0.92));
      animation:_drop 0.82s cubic-bezier(0.23,1,0.32,1) forwards;
    }
    @keyframes _drop {
      from{opacity:0; transform:translateY(-65px) scale(0.82);}
      to  {opacity:1; transform:translateY(0) scale(1);}
    }

    /* ── Interior parchment page ── */
    #ls-i-back {
      position:absolute; inset:0;
      background:linear-gradient(108deg,#F8F2E0 0%,#F3EAD5 55%,#EDE4C8 100%);
      border-radius:2px 6px 6px 2px; overflow:hidden;
      display:flex; align-items:center; justify-content:center;
    }
    #ls-i-back-spine {
      position:absolute; left:0; top:0; bottom:0; width:14px;
      background:linear-gradient(to right,#5A3010 0%,#9A6020 40%,#C08030 50%,#9A6020 70%,#5A3010 100%);
    }
    #ls-i-back-content {
      display:flex; flex-direction:column; align-items:center;
      gap:0.55rem; padding-left:0.75rem; text-align:center;
    }
    /* Step 3 — text rises after cover opens */
    #ls-i-back-title {
      font-family:'Playfair Display',serif; line-height:1.15;
      opacity:0; animation:_rise 0.55s ease 2.2s forwards;
    }
    #ls-i-back-title .t { display:block; font-size:1.9rem; font-weight:700; color:#8B5E1A; }
    #ls-i-back-title .i { display:block; font-size:1.6rem; font-style:italic; font-weight:500; color:#2A1408; }
    #ls-i-back-rule {
      width:52px; height:1px;
      background:linear-gradient(to right,transparent,#C8861E,transparent);
      opacity:0; animation:_rise 0.45s ease 2.32s forwards;
    }
    #ls-i-back-sub {
      font-family:'Source Serif 4',serif; font-size:0.75rem; font-style:italic; color:#9A7A50;
      opacity:0; animation:_rise 0.45s ease 2.44s forwards;
    }
    @keyframes _rise { from{opacity:0;transform:translateY(8px);} to{opacity:1;transform:translateY(0);} }

    /* ── Front leather cover — Step 2 flip ── */
    #ls-i-cover {
      position:absolute; inset:0;
      transform-origin:left center; transform-style:preserve-3d;
      animation:_flip 1.2s cubic-bezier(0.4,0,0.2,1) 1.0s forwards;
    }
    @keyframes _flip { 0%{transform:rotateY(0deg);} 100%{transform:rotateY(-162deg);} }

    #ls-i-cover-front {
      position:absolute; inset:0;
      backface-visibility:hidden; -webkit-backface-visibility:hidden;
      background:linear-gradient(158deg,#4A200A 0%,#2C1005 55%,#1C0803 100%);
      border-radius:2px 6px 6px 2px; overflow:hidden;
      box-shadow:6px 0 24px rgba(20,8,2,0.6), inset 0 0 80px rgba(0,0,0,0.2);
    }
    #ls-i-cover-front::before {
      content:''; position:absolute; inset:0; pointer-events:none;
      background:
        repeating-linear-gradient( 45deg,transparent 0,transparent 3px,rgba(255,255,255,0.013) 3px,rgba(255,255,255,0.013) 4px),
        repeating-linear-gradient(-45deg,transparent 0,transparent 3px,rgba(255,255,255,0.013) 3px,rgba(255,255,255,0.013) 4px);
    }
    #ls-i-cover-front::after {
      content:''; position:absolute; inset:0; pointer-events:none;
      background:radial-gradient(ellipse 60% 30% at 70% 15%,rgba(200,134,30,0.07),transparent);
    }

    #ls-i-spine {
      position:absolute; left:0; top:0; bottom:0; width:16px;
      background:linear-gradient(to right,#0A0300 0%,#1C0803 35%,#2E1005 55%,#1C0803 75%,#0A0300 100%);
      box-shadow:inset -2px 0 5px rgba(0,0,0,0.5);
    }
    #ls-i-spine::before {
      content:''; position:absolute; left:6px; top:20px; bottom:20px; width:1px;
      background:repeating-linear-gradient(to bottom,rgba(200,134,30,0.45) 0,rgba(200,134,30,0.45) 4px,transparent 4px,transparent 9px);
    }

    #ls-i-body {
      position:absolute; inset:0; left:18px;
      display:flex; flex-direction:column; align-items:center; justify-content:center;
      gap:0.7rem; padding:2.5rem 1.5rem;
    }
    #ls-i-frame {
      position:absolute; inset:10px; left:22px;
      border:1px solid rgba(200,134,30,0.22); pointer-events:none;
    }
    #ls-i-frame::before { content:''; position:absolute; inset:5px; border:1px solid rgba(200,134,30,0.1); }
    #ls-i-frame::after  { content:'◆'; position:absolute; top:-7px; left:50%; transform:translateX(-50%); font-size:0.5rem; color:rgba(200,134,30,0.45); }

    #ls-i-ornament { font-size:0.95rem; color:rgba(200,134,30,0.75); letter-spacing:0.3em; }
    #ls-i-title { font-family:'Playfair Display',serif; text-align:center; line-height:1.15; }
    #ls-i-title .t { display:block; font-size:2.6rem; font-weight:700; color:#E8A830; }
    #ls-i-title .i { display:block; font-size:2.2rem; font-style:italic; font-weight:500; color:#FAF4E8; }
    #ls-i-deco { width:55px; height:1px; background:linear-gradient(to right,transparent,#C8861E,transparent); }
    #ls-i-desc { font-family:'Source Serif 4',serif; font-size:0.72rem; color:#B8A484; font-style:italic; letter-spacing:0.06em; text-align:center; opacity:0.75; }
    #ls-i-year { font-family:'JetBrains Mono',monospace; font-size:0.55rem; color:rgba(200,134,30,0.4); letter-spacing:0.22em; margin-top:0.3rem; }

    #ls-i-cover-inside {
      position:absolute; inset:0;
      backface-visibility:hidden; -webkit-backface-visibility:hidden;
      transform:rotateY(180deg);
      background:linear-gradient(108deg,#EAE0C8,#E4D8B8);
      border-radius:6px 2px 2px 6px;
    }

    /* Caption fades in early */
    #ls-i-caption {
      font-family:'JetBrains Mono',monospace; font-size:0.57rem;
      letter-spacing:0.22em; text-transform:uppercase;
      color:rgba(200,134,30,0.35);
      opacity:0; animation:_rise 0.5s ease 0.35s forwards;
    }
    </style>

    <div id="ls-intro">
      <div id="ls-i-glow"></div>
      <div id="ls-i-scene">
        <div id="ls-i-book">
          <div id="ls-i-back">
            <div id="ls-i-back-spine"></div>
            <div id="ls-i-back-content">
              <div id="ls-i-back-title">
                <span class="t">Legacy</span><em class="i">Scribe</em>
              </div>
              <div id="ls-i-back-rule"></div>
              <div id="ls-i-back-sub">Every family has a story worth keeping</div>
            </div>
          </div>
          <div id="ls-i-cover">
            <div id="ls-i-cover-front">
              <div id="ls-i-spine"></div>
              <div id="ls-i-frame"></div>
              <div id="ls-i-body">
                <div id="ls-i-ornament">✦ · ✦</div>
                <div id="ls-i-title">
                  <span class="t">Legacy</span><em class="i">Scribe</em>
                </div>
                <div id="ls-i-deco"></div>
                <div id="ls-i-desc">A Family Memory Journal</div>
                <div id="ls-i-year">· 2025 ·</div>
              </div>
            </div>
            <div id="ls-i-cover-inside"></div>
          </div>
        </div>
      </div>
      <div id="ls-i-caption">Family Memory Agent</div>
    </div>
    """)

    # ── Header ──
    gr.HTML("""
    <div class="ls-header">
      <div class="ls-wordmark">
        <span class="ls-wordmark-legacy">Legacy</span><span class="ls-wordmark-scribe">Scribe</span><span class="ls-wordmark-dot">.</span>
      </div>
      <div class="ls-deco-line"></div>
      <div class="ls-tagline">Every family has a story worth keeping</div>
      <div class="ls-badges">
        <span class="ls-badge">Off the Grid</span>
        <span class="ls-badge">Qwen3.5 · 9B</span>
        <span class="ls-badge">LoRA Fine-tuned</span>
        <span class="ls-badge">Nepali · English</span>
        <span class="ls-badge">5-Agent Pipeline</span>
        <span class="ls-badge">Build Small 2025</span>
      </div>
    </div>
    """)

    # ── Main two-column layout ──
    with gr.Row(elem_classes=["ls-main"]):

        # Left — conversation
        with gr.Column(elem_classes=["ls-conversation"]):
            gr.HTML('<div class="ls-section-label">Your memory</div>')
            chat_display   = gr.HTML(build_chat_html([]))
            notes_display  = gr.HTML("")

            with gr.Column(elem_classes=["ls-input-wrap"]):
                user_input = gr.Textbox(
                    placeholder="Tell me about a person, a festival, a meal, a place — anything you remember…",
                    lines=3,
                    show_label=False,
                    container=False,
                )

            with gr.Row(elem_classes=["ls-btn-row"]):
                submit_btn = gr.Button("Share this memory →", elem_classes=["ls-btn", "ls-btn-primary"])
                clear_btn  = gr.Button("Start over",          elem_classes=["ls-btn", "ls-btn-secondary"])

            thinking = gr.HTML(
                '<div class="ls-thinking" style="display:none">'
                '<div class="ls-dot"></div><div class="ls-dot"></div><div class="ls-dot"></div>'
                '&nbsp;Listening and writing…</div>'
            )

        # Right — memory book
        with gr.Column(elem_classes=["ls-book"]):
            gr.HTML('<div class="ls-section-label">Your memory book</div>')
            page_display = gr.HTML(build_page_html("", 1))

            def export_memory(state):
                if not state["chapter"]:
                    return gr.update(visible=False)
                content  = "# My Memory Book\n\n"
                content += f"## Chapter {state['chapter_num']}\n\n"
                content += state["chapter"] + "\n\n---\n\n"
                content += "### Memory fragments\n\n"
                content += "\n".join(f"- {n}" for n in state["all_notes"])
                import tempfile, pathlib
                path = pathlib.Path(tempfile.gettempdir()) / "legacyscribe_memory.txt"
                path.write_text(content, encoding="utf-8")
                return gr.update(visible=True, value=str(path))

            dl_btn    = gr.Button("↓ Download memory book", elem_classes=["ls-dl-btn"])
            dl_output = gr.File(visible=False, label="Your memory book")
            dl_btn.click(fn=export_memory, inputs=[state], outputs=[dl_output])

    # ── Footer ──
    gr.HTML("""
    <div class="ls-footer">
      LegacyScribe &nbsp;·&nbsp; Build Small Hackathon 2025 &nbsp;·&nbsp;
      Fine-tuned Qwen3.5-9B &nbsp;·&nbsp; Runs fully offline via llama.cpp
    </div>
    """)

    # ── Handlers ──
    def on_submit(user_text, state):
        if not user_text.strip():
            return (state,
                    build_chat_html(state["turns"]),
                    "",
                    build_page_html(state["chapter"], state["chapter_num"]),
                    "")
        new_state, question, arc, chapter, notes = process_turn(user_text, state)
        return (new_state,
                build_chat_html(new_state["turns"]),
                build_notes_html(notes),
                build_page_html(chapter, new_state["chapter_num"]),
                "")

    def on_clear():
        s = init_state()
        return s, build_chat_html([]), "", build_page_html("", 1), ""

    submit_btn.click(
        fn=on_submit,
        inputs=[user_input, state],
        outputs=[state, chat_display, notes_display, page_display, user_input],
    )
    user_input.submit(
        fn=on_submit,
        inputs=[user_input, state],
        outputs=[state, chat_display, notes_display, page_display, user_input],
    )
    clear_btn.click(
        fn=on_clear,
        inputs=[],
        outputs=[state, chat_display, notes_display, page_display, user_input],
    )

if __name__ == "__main__":
    demo.launch()