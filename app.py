"""FRAME — AI Film Pre-Production Agent (Streamlit UI).

Run:  streamlit run app.py

State model: `st.session_state.project` (a Pydantic `Project`) is the single source of
truth. The chat transcript only stores text. Sidebar / input widgets are mirrored into
the project at the start of every run (`pull_widgets`) and refreshed from it after an
agent turn (`sync_widgets` flag -> `push_widgets`).
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import streamlit as st

from agent import NEXT_LABEL, FrameAgent
from export_utils import safe_filename
from llm_provider import LLMConfig, create_provider, load_env_file
from sample_project import sample_brief, sample_full, sample_with_flaws
from schemas import ArtifactStatus, BudgetLevel, Intent, Project, Severity, Stage

load_env_file(Path(__file__).parent / ".env")

st.set_page_config(page_title="FRAME — AI Film Pre-Production", page_icon="🎬", layout="wide")

# --------------------------------------------------------------------------- #
# Styling (scoped to .frame-* classes plus a few Streamlit widgets)
# --------------------------------------------------------------------------- #
st.markdown(
    """
<style>
:root { --frame-accent:#d9a441; --frame-panel:#1a1d24; --frame-line:#2a2e38; --frame-muted:#9a9a9a; }
.block-container { padding-top: 1.6rem; max-width: 1500px; }
h1, h2, h3 { font-family: Georgia, 'Times New Roman', serif; letter-spacing: .01em; }
.frame-brand { font-family: Georgia, serif; font-size: 2.1rem; letter-spacing: .28em; color: var(--frame-accent); margin: 0; }
.frame-tag { color: var(--frame-muted); font-size: .9rem; margin-top: -.2rem; }
.frame-card { background: var(--frame-panel); border: 1px solid var(--frame-line); border-radius: 10px; padding: 1rem 1.2rem; }
.frame-chip { display:inline-block; padding: .1rem .6rem; border-radius: 999px; font-size:.78rem; border:1px solid var(--frame-line); color:var(--frame-muted); margin-right:.3rem; }
.frame-chip.ok { color:#7ec699; border-color:#2f5a40; } .frame-chip.bad { color:#e08a8a; border-color:#6a3434; }
.frame-chip.sample { color:var(--frame-accent); border-color:#6b5320; }
.frame-step { display:flex; justify-content:space-between; padding:.18rem 0; font-size:.9rem; border-bottom:1px dashed var(--frame-line); }
.frame-hero h3 { margin-top:0; color: var(--frame-accent); }
div[data-testid="stTabs"] button[role="tab"] { font-size: .95rem; }
div[data-testid="stSidebar"] { border-right: 1px solid var(--frame-line); }
@media (max-width: 900px) { .frame-brand { font-size: 1.5rem; } .block-container { padding: 1rem .6rem; } }
</style>
""",
    unsafe_allow_html=True,
)

BUDGET_LABEL = {"LOW": "Бага", "MEDIUM": "Дунд", "HIGH": "Өндөр"}
STAGE_TITLE = {
    Stage.IDEA: "Идея", Stage.SCRIPT: "Сценар", Stage.SCENES: "Дүр зургийн задлал",
    Stage.SHOTS: "Шотын жагсаалт", Stage.PLAN: "Үйлдвэрлэлийн төлөвлөгөө", Stage.VALIDATION: "Шалгалт",
}
STATUS_ICON = {ArtifactStatus.FRESH: "✅", ArtifactStatus.OUTDATED: "⚠️", ArtifactStatus.EMPTY: "⬜"}


# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #
def init_state() -> None:
    ss = st.session_state
    if "project" in ss:
        return
    ss.project = Project()
    ss.chat = []
    ss.llm_config = LLMConfig.from_env()
    ss.provider = create_provider(ss.llm_config)
    ss.llm_status = ss.provider.check_status()
    ss.last_turn = None
    ss.export = None
    ss.sync_widgets = True


def push_widgets(p: Project) -> None:
    """Project -> widget state (only before the widgets are instantiated)."""
    ss = st.session_state
    ss.w_name, ss.w_genre, ss.w_duration = p.name, p.genre, float(p.duration_minutes)
    ss.w_max_loc, ss.w_max_act = p.max_locations or 0, p.max_actors or 0
    ss.w_budget, ss.w_style, ss.w_constraints = p.budget.value, p.visual_style, p.constraints
    ss.w_equipment, ss.w_language = "\n".join(p.equipment), p.language
    ss.w_idea, ss.w_decisions = p.idea, "\n".join(p.approved_decisions)


def pull_widgets(p: Project) -> None:
    """Widget state -> Project (start of every run, so project is always current)."""
    g = st.session_state.get
    if g("w_name") is not None:
        p.name = g("w_name").strip() or "Нэргүй төсөл"
        p.genre = g("w_genre", "").strip()
        p.duration_minutes = float(g("w_duration", 3.0))
        p.max_locations = int(g("w_max_loc", 0)) or None
        p.max_actors = int(g("w_max_act", 0)) or None
        p.budget = BudgetLevel(g("w_budget", "LOW"))
        p.visual_style = g("w_style", "").strip()
        p.constraints = g("w_constraints", "").strip()
        p.equipment = [e.strip() for e in g("w_equipment", "").splitlines() if e.strip()]
        p.language = g("w_language", "mn")
    if g("w_idea") is not None:
        p.idea = g("w_idea", "").strip()
        p.approved_decisions = [d.strip() for d in g("w_decisions", "").splitlines() if d.strip()]


def load_project(project: Project, note: str) -> None:
    """on_click callback: replace the project and reset the chat."""
    ss = st.session_state
    ss.project = project
    ss.chat = [{"role": "assistant", "content": note}]
    ss.last_turn = None
    ss.export = None
    ss.sync_widgets = True


def approve_decision(text: str) -> None:
    """on_click callback: add an AI-suggested decision to the approved list."""
    ss = st.session_state
    pull_widgets(ss.project)
    if text not in ss.project.approved_decisions:
        ss.project.approved_decisions.append(text)
    ss.sync_widgets = True


def recheck_model() -> None:
    ss = st.session_state
    ss.provider = create_provider(ss.llm_config)
    ss.llm_status = ss.provider.check_status()


def apply_model_settings() -> None:
    """Apply the Ollama settings widgets (only shown when the Ollama provider is active)."""
    ss = st.session_state
    ss.llm_config = replace(
        ss.llm_config,
        base_url=ss.w_base_url.strip().rstrip("/") or ss.llm_config.base_url,
        model=ss.w_model.strip() or ss.llm_config.model,
        timeout_seconds=float(ss.w_timeout))
    recheck_model()


# --------------------------------------------------------------------------- #
# Agent invocation
# --------------------------------------------------------------------------- #
def run_agent(message: str, *, intent: Intent | None = None, args: dict | None = None,
              label: str | None = None) -> None:
    ss = st.session_state
    ss.chat.append({"role": "user", "content": label or message})
    agent = FrameAgent(ss.provider)
    with st.status("Төслийг шинжилж байна…", expanded=True) as box:
        def on_status(text: str) -> None:
            box.write(f"• {text}…")
            box.update(label=f"{text}…")

        turn = agent.handle(ss.project, message, forced_intent=intent, forced_args=args,
                            history=ss.chat[:-1], on_status=on_status)
        box.update(label="Дууслаа" if turn.ok else "Анхааруулга / алдаа", expanded=False,
                   state="complete" if turn.ok else "error")
    ss.chat.append({
        "role": "assistant", "content": turn.reply, "ok": turn.ok,
        "steps": [{"tool": s.tool, "ok": s.ok, "sec": round(s.elapsed_sec, 1)} for s in turn.steps],
    })
    ss.last_turn = turn
    if turn.export_markdown:
        ss.export = {"markdown": turn.export_markdown, "filename": turn.export_filename}
    ss.sync_widgets = True
    st.rerun()


def action_button(label: str, intent: Intent, *, key: str, args: dict | None = None,
                  primary: bool = False, chat_label: str | None = None) -> None:
    if st.button(label, key=key, type="primary" if primary else "secondary"):
        run_agent("", intent=intent, args=args, label=chat_label or f"▶ {label}")


def stale_banner(stage: Stage) -> None:
    if st.session_state.project.get_status(stage) == ArtifactStatus.OUTDATED:
        st.warning("Дээд түвшний өгөгдөл өөрчлөгдсөн тул энэ хэсэг **хуучирсан**. Дахин үүсгэнэ үү.")


def provenance_chip(stage: Stage) -> None:
    origin = st.session_state.project.provenance.get(stage, "")
    if origin == "sample":
        st.markdown('<span class="frame-chip sample">SAMPLE — статик жишээ, AI үр дүн биш</span>',
                    unsafe_allow_html=True)
    elif origin.startswith("model"):
        st.markdown(f'<span class="frame-chip">AI · {origin.split(":", 1)[1]}</span>', unsafe_allow_html=True)
    elif origin.startswith("deterministic"):
        st.markdown(f'<span class="frame-chip">{origin}</span>', unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
def render_sidebar(p: Project) -> None:
    ss = st.session_state
    with st.sidebar:
        st.markdown('<p class="frame-brand">FRAME</p><p class="frame-tag">AI Film Pre-Production Agent</p>',
                    unsafe_allow_html=True)
        st.divider()
        st.subheader("Төсөл")
        st.text_input("Төслийн нэр", key="w_name")
        st.text_input("Жанр", key="w_genre", placeholder="жишээ: нууцлаг триллер")
        st.number_input("Хугацаа (минут)", min_value=0.5, max_value=30.0, step=0.5, key="w_duration")
        c1, c2 = st.columns(2)
        c1.number_input("Байршил (дээд)", min_value=0, max_value=20, key="w_max_loc", help="0 = хязгааргүй")
        c2.number_input("Жүжигчин (дээд)", min_value=0, max_value=30, key="w_max_act", help="0 = хязгааргүй")
        st.selectbox("Төсвийн түвшин", list(BUDGET_LABEL), format_func=BUDGET_LABEL.get, key="w_budget")
        st.text_input("Визуал хэв маяг", key="w_style")
        st.text_area("Хязгаарлалт", key="w_constraints", height=80)
        st.text_area("Байгаа тоног төхөөрөмж (мөр бүрт нэг)", key="w_equipment", height=90)
        st.selectbox("Үр дүнгийн хэл", ["mn", "en"],
                     format_func=lambda v: "Монгол" if v == "mn" else "English", key="w_language")

        st.divider()
        st.subheader("AI загвар")
        status = ss.llm_status
        cfg = ss.llm_config
        chip = "ok" if status.available else "bad"
        st.markdown(f'<span class="frame-chip {chip}">{"бэлэн" if status.available else "боломжгүй"}</span>'
                    f'<span class="frame-chip">{cfg.provider} · {ss.provider.model or "—"}</span>', unsafe_allow_html=True)
        if not status.available:
            st.warning(f"{status.message}\n\n**Demo горим:** зөвхөн дүрэмд суурилсан tool-ууд "
                       "(шалгалт, төлөвлөгөө, экспорт) ажиллана.")
        with st.expander("Загварын тохиргоо"):
            if cfg.provider == "ollama":
                st.text_input("Ollama URL", value=cfg.base_url, key="w_base_url")
                st.text_input("Загварын нэр", value=cfg.model, key="w_model")
                st.number_input("Timeout (сек)", min_value=10, max_value=1800, value=int(cfg.timeout_seconds),
                                key="w_timeout")
                st.button("Хэрэглэх & дахин шалгах", on_click=apply_model_settings)
            else:
                st.caption(f"Провайдер: **{cfg.provider}**. Загвар, түлхүүрийг `.env` (GEMINI_MODEL, "
                           "GEMINI_API_KEY)-ээс уншина; түлхүүр энд харагдахгүй. `.env`-г өөрчилвөл "
                           "Streamlit-ийг дахин ачаална уу.")
        if cfg.provider == "gemini":
            st.caption("⚠️ Gemini free tier-д илгээсэн агуулга Google-ийн бүтээгдэхүүнийг сайжруулахад "
                       "ашиглагдаж болно. Нууц агуулга бүү илгээгээрэй.")
        st.button("Холболтыг дахин шалгах", on_click=recheck_model)

        st.divider()
        st.subheader("Явц")
        done = sum(1 for s in Stage if p.is_fresh(s))
        st.progress(done / len(Stage))
        for stage in Stage:
            stt = p.get_status(stage)
            suffix = " · sample" if p.provenance.get(stage) == "sample" else ""
            st.markdown(f'<div class="frame-step"><span>{STAGE_TITLE[stage]}</span>'
                        f'<span>{STATUS_ICON[stt]} {stt.value}{suffix}</span></div>', unsafe_allow_html=True)

        st.divider()
        st.subheader("Жишээ төсөл")
        st.button("🎬 The Last Bus — зөвхөн brief", width="stretch", on_click=load_project, args=(
            sample_brief(), "«The Last Bus» brief ачааллаа. Миний AI-аар бүх алхмыг ажиллуулж үзэхийн тулд "
                            "«Санаа хөгжүүлэх» эсвэл «Сценар бичих» гэж хэлнэ үү."))
        st.button("📦 The Last Bus — бүрэн demo (static)", width="stretch", on_click=load_project, args=(
            sample_full(), "Бүрэн **статик жишээ** ачааллаа (AI үр дүн биш — SAMPLE гэж тэмдэглэсэн). "
                           "«Шалгах», «Багц экспортлох» зэргийг загваргүйгээр туршиж болно."))
        st.button("🧪 Demo: алдаатай жишээ", width="stretch", on_click=load_project, args=(
            sample_with_flaws(), "Зориудаар алдаа оруулсан **статик жишээ** ачааллаа. "
                                 "«Validation» таб дээр шалгалт ажиллуулж алдаануудыг харна уу."))
        st.download_button("💾 Төслийг JSON-оор хадгалах", data=p.model_dump_json(indent=2),
                           file_name=safe_filename(p.name, ".json"), mime="application/json", width="stretch")


# --------------------------------------------------------------------------- #
# Chat column
# --------------------------------------------------------------------------- #
def render_chat() -> None:
    ss = st.session_state
    st.subheader("Туслах")
    box = st.container(height=520, border=True)
    with box:
        if not ss.chat:
            with st.chat_message("assistant"):
                st.markdown("Сайн байна уу! Би **FRAME** — таны киноны бэлтгэл ажлын туслах.\n\n"
                            "Киноны санаагаа бичээрэй, эсвэл зүүн талын **жишээ төсөл** сонгоно уу. "
                            "Би санаа → сценар → дүр зураг → шот → төлөвлөгөө → шалгалт → экспорт "
                            "дарааллаар хамт ажиллана.")
        for msg in ss.chat:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("steps"):
                    with st.expander("Ажилласан tool-ууд"):
                        for s in msg["steps"]:
                            st.write(f"{'✅' if s['ok'] else '❌'} `{s['tool']}` · {s['sec']} сек")

    turn = ss.last_turn
    if turn and turn.suggested_intent and turn.suggested_intent != Intent.GENERAL_QUESTION:
        action_button(f"▶ Дараагийн алхам: {NEXT_LABEL[turn.suggested_intent]}", turn.suggested_intent,
                      key="next_action", primary=True)

    if prompt := st.chat_input("Киноны санаа эсвэл команд бичнэ үү…"):
        run_agent(prompt)


# --------------------------------------------------------------------------- #
# Result tabs
# --------------------------------------------------------------------------- #
def empty_state() -> None:
    st.markdown(
        '<div class="frame-card frame-hero"><h3>Шинэ төслөө эхлүүлье</h3>'
        "<p>1. Зүүн талд киноны <b>санаагаа</b> бичих эсвэл <b>жишээ төсөл</b> сонгох<br>"
        "2. Туслах сценар, дүр зураг, шот, төлөвлөгөөг алхам алхмаар үүсгэнэ<br>"
        "3. Шалгалт хийж, <b>Markdown багц</b> татаж авна</p></div>",
        unsafe_allow_html=True)
    st.write("")


def tab_idea(p: Project) -> None:
    st.text_area("Киноны санаа (premise)", key="w_idea", height=110,
                 placeholder="Жишээ: Хоосон автобусны зогсоолд хүлээж буй охин ирээдүйг 10 секундээр харна…")
    st.text_area("Батлагдсан бүтээлч шийдвэрүүд (мөр бүрт нэг) — AI эдгээрийг хадгална", key="w_decisions",
                 height=90)
    action_button("Санаа хөгжүүлэх", Intent.DEVELOP_IDEA, key="btn_idea", primary=True)
    provenance_chip(Stage.IDEA)

    notes = p.idea_notes
    if notes:
        st.markdown(f"**Дүгнэлт:** {notes.summary}")
        cols = st.columns(2)
        with cols[0]:
            st.markdown("**Логлайны санал**")
            for opt in notes.logline_options:
                st.markdown(f"- {opt}")
            st.markdown("**Сэдвүүд:** " + ", ".join(notes.themes))
        with cols[1]:
            st.markdown("**Тодруулах асуулт**")
            for q in notes.story_questions:
                st.markdown(f"- {q}")
        if notes.suggested_decisions:
            st.markdown("**Санал болгож буй шийдвэрүүд**")
            for i, d in enumerate(notes.suggested_decisions):
                c1, c2 = st.columns([5, 1])
                c1.write(d)
                c2.button("Батлах", key=f"approve_{i}", on_click=approve_decision, args=(d,),
                          disabled=d in p.approved_decisions)

    if p.logline:
        st.divider()
        st.subheader("Story")
        provenance_chip(Stage.SCRIPT)
        stale_banner(Stage.SCRIPT)
        st.markdown(f"**Логлайн:** {p.logline}")
        st.markdown(f"**Синопсис:** {p.synopsis}")
        st.markdown("**Дүрүүд**")
        st.dataframe([{"Нэр": c.name, "Үүрэг": c.role, "Тайлбар": c.description,
                       "Continuity дэлгэрэнгүй": ", ".join(c.details)} for c in p.characters],
                     hide_index=True, width="stretch")


def instruction_input(key: str) -> dict:
    text = st.text_input("Нэмэлт заавар (заавал биш)", key=key,
                         help="Дахин үүсгэхдээ юуг өөрчлөхийг энд бичнэ. Батлагдсан шийдвэрүүд хадгалагдана.")
    return {"instructions": text.strip()} if text.strip() else {}


def tab_script(p: Project) -> None:
    args = instruction_input("w_instr_script")
    c1, c2 = st.columns(2)
    with c1:
        action_button("Сценар үүсгэх / дахин үүсгэх" if p.script_scenes else "Сценар үүсгэх",
                      Intent.BUILD_SCRIPT, key="btn_script", args=args, primary=not p.script_scenes)
    with c2:
        action_button("Дүр зургийг задлах", Intent.BREAKDOWN_SCENES, key="btn_scenes",
                      args=args if p.script_scenes else None)

    st.markdown("##### Сценар")
    stale_banner(Stage.SCRIPT)
    if not p.script_scenes:
        st.info("Сценар хараахан үүсээгүй байна. Эхлээд санаагаа оруулаад «Сценар үүсгэх» дарна уу.")
    else:
        provenance_chip(Stage.SCRIPT)
        for s in p.script_scenes:
            with st.expander(f"Дүр зураг {s.number} — {s.heading}", expanded=s.number == 1):
                st.text(s.text)

    st.markdown("##### Дүр зургийн задлал")
    stale_banner(Stage.SCENES)
    if not p.scenes:
        st.info("Дүр зургийн задлал хараахан хийгдээгүй байна.")
    else:
        provenance_chip(Stage.SCENES)
        m1, m2, m3 = st.columns(3)
        m1.metric("Дүр зураг", len(p.scenes))
        m2.metric("Нийт хугацаа", f"{p.total_scene_seconds:g} сек", f"зорилт {p.duration_seconds:g} сек",
                  delta_color="off")
        m3.metric("Байршил", len({s.location.casefold() for s in p.scenes}))
        st.dataframe([{
            "#": s.number, "Байршил": s.location, "Цаг": s.time_of_day.value,
            "Дүрүүд": ", ".join(s.characters), "Props": ", ".join(s.props), "Үйлдэл": s.action,
            "Сек": s.estimated_duration_sec} for s in p.scenes], hide_index=True, width="stretch")


def tab_shots(p: Project) -> None:
    args = instruction_input("w_instr_shots")
    action_button("Шотын жагсаалт үүсгэх / дахин үүсгэх" if p.shots else "Шотын жагсаалт үүсгэх",
                  Intent.CREATE_SHOT_LIST, key="btn_shots", args=args, primary=not p.shots)
    stale_banner(Stage.SHOTS)
    if not p.shots:
        st.info("Шотын жагсаалт хараахан үүсээгүй байна. Эхлээд дүр зургийг задлах шаардлагатай.")
        return
    provenance_chip(Stage.SHOTS)
    m1, m2 = st.columns(2)
    m1.metric("Шот", len(p.shots))
    m2.metric("Нийт хугацаа", f"{sum(s.estimated_duration_sec for s in p.shots):g} сек")
    st.dataframe([{
        "#": s.number, "Scene": s.scene_number, "Type": s.shot_type.value, "Angle": s.camera_angle.value,
        "Movement": s.movement.value, "Тайлбар": s.description, "Дуу": s.audio_notes,
        "Сек": s.estimated_duration_sec} for s in p.shots], hide_index=True, width="stretch")


def tab_plan(p: Project) -> None:
    action_button("Төлөвлөгөө гаргах / шинэчлэх", Intent.PLAN_PRODUCTION, key="btn_plan", primary=not p.plan)
    stale_banner(Stage.PLAN)
    if not p.plan:
        st.info("Төлөвлөгөө хараахан гараагүй байна. Шотын жагсаалт бэлэн болсны дараа ажиллана.")
        return
    plan = p.plan
    provenance_chip(Stage.PLAN)
    st.caption("Доорх зүйлс нь дүрэмд суурилсан ТААМАГЛАЛ. Үнэ, зардал тооцоогүй.")
    m1, m2, m3 = st.columns(3)
    m1.metric("Зураг авалтын өдөр", plan.shoot_days)
    m2.metric("Нийт (таамаг)", f"~{plan.total_minutes} мин")
    m3.metric("Props", len(plan.props))
    st.markdown("##### Хуваарь")
    st.dataframe([{"Өдөр": b.day, "Дараалал": b.order, "Блок": b.label,
                   "Дүр зураг": ", ".join(map(str, b.scene_numbers)), "Мин (таамаг)": b.estimated_minutes,
                   "Тэмдэглэл": b.notes} for b in plan.schedule], hide_index=True, width="stretch")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Тоног төхөөрөмж")
        mark = {True: "✅ байгаа", False: "❓ жагсаалтад алга", None: "—"}
        st.dataframe([{"Нэр": e.name, "Шалтгаан": e.reason, "Төлөв": mark[e.available]}
                      for e in plan.equipment], hide_index=True, width="stretch")
    with c2:
        st.markdown("##### Props")
        st.dataframe([{"Нэр": x.name, "Дүр зураг": ", ".join(map(str, x.scene_numbers))}
                      for x in plan.props], hide_index=True, width="stretch")
    st.markdown("##### Байршил")
    st.dataframe([{"Байршил": x.name, "Дүр зураг": ", ".join(map(str, x.scene_numbers)),
                   "Цаг": x.time_of_day} for x in plan.locations], hide_index=True, width="stretch")
    st.info("**Таамаглал**\n\n" + "\n".join(f"- {a}" for a in plan.assumptions))


def tab_validation(p: Project) -> None:
    use_model = st.checkbox("AI-ийн семантик шалгалт нэмэх (удаан; зөвхөн санал, баталгаагүй)",
                            key="w_use_model", value=False)
    action_button("Шалгалт ажиллуулах", Intent.CHECK_CONTINUITY, key="btn_check",
                  args={"use_model": use_model}, primary=not p.is_fresh(Stage.VALIDATION))
    stale_banner(Stage.VALIDATION)
    if not p.is_fresh(Stage.VALIDATION):
        st.info("Шалгалт хараахан ажиллаагүй байна. Дүр зургийн задлал (болон шот) бэлэн болсны дараа ажиллуулна уу.")
    else:
        provenance_chip(Stage.VALIDATION)
        counts = {sev: sum(1 for i in p.issues if i.severity == sev) for sev in Severity}
        m1, m2, m3 = st.columns(3)
        m1.metric("Алдаа", counts[Severity.ERROR])
        m2.metric("Анхааруулга", counts[Severity.WARNING])
        m3.metric("Тэмдэглэл", counts[Severity.INFO])
        if not p.issues:
            st.success("Дүрмийн шалгалтаар асуудал илрээгүй.")
        color = {Severity.ERROR: "red", Severity.WARNING: "orange", Severity.INFO: "blue"}
        for issue in sorted(p.issues, key=lambda i: list(Severity).index(i.severity)):
            tag = "дүрмэн шалгалт" if issue.source == "deterministic" else "AI санал · баталгаагүй"
            where = f" · дүр зураг {issue.scene_number}" if issue.scene_number else ""
            where += f" · шот {issue.shot_number}" if issue.shot_number else ""
            with st.container(border=True):
                st.markdown(f":{color[issue.severity]}[**{issue.severity.value}**] `{issue.code}` "
                            f"_{tag}{where}_")
                st.write(issue.message)
                if issue.suggestion:
                    st.caption(f"Санал: {issue.suggestion}")

    st.divider()
    st.markdown("##### Экспорт")
    action_button("Үйлдвэрлэлийн багц бэлтгэх (Markdown)", Intent.EXPORT_PACKAGE, key="btn_export")
    export = st.session_state.export
    if export:
        st.download_button("⬇️ Markdown татаж авах", data=export["markdown"], file_name=export["filename"],
                           mime="text/markdown", type="primary")
        with st.expander("Урьдчилан харах"):
            st.markdown(export["markdown"])


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    init_state()
    ss = st.session_state
    if ss.pop("sync_widgets", False):
        push_widgets(ss.project)
    pull_widgets(ss.project)
    p: Project = ss.project

    render_sidebar(p)

    st.markdown('<p class="frame-brand">FRAME</p><p class="frame-tag">Киноны бэлтгэл ажлын AI туслах · '
                  "санаа → сценар → шот → төлөвлөгөө → экспорт</p>", unsafe_allow_html=True)
    if not ss.llm_status.available:
        st.warning("**Demo горим** — AI загвар холбогдоогүй. Дүрэмд суурилсан tool-ууд (шалгалт, төлөвлөгөө, "
                   "экспорт) ажиллана. Загвар шаардсан алхмууд тодорхой алдаа харуулна. "
                   f"({ss.llm_status.message})")

    chat_col, work_col = st.columns([5, 7], gap="large")
    with chat_col:
        render_chat()
    with work_col:
        if p.is_blank and not ss.chat:
            empty_state()
        t1, t2, t3, t4, t5 = st.tabs(
            ["Idea & Story", "Script & Scenes", "Shot List", "Production Plan", "Validation"])
        with t1:
            tab_idea(p)
        with t2:
            tab_script(p)
        with t3:
            tab_shots(p)
        with t4:
            tab_plan(p)
        with t5:
            tab_validation(p)


main()
