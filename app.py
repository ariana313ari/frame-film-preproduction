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

import ui_theme as ui
from agent import NEXT_LABEL, FrameAgent, next_intent
from export_utils import safe_filename
from llm_provider import LLMConfig, apply_secrets, create_provider, load_env_file
from sample_project import sample_brief, sample_full, sample_with_flaws
from schemas import ArtifactStatus, BudgetLevel, Intent, Project, Severity, Stage

load_env_file(Path(__file__).parent / ".env")
try:  # Streamlit Cloud "Secrets" / local .streamlit/secrets.toml (names only; values never shown)
    apply_secrets(st.secrets)
except FileNotFoundError:  # no secrets configured: fine, the status panel explains what is missing
    pass

# Brand assets, resolved relative to this file so `streamlit run app.py` works from any directory.
ASSETS = Path(__file__).parent / "assets"
LOGO = ASSETS / "frame_logo.png"
ICON = ASSETS / "frame_app_icon.png"

st.set_page_config(page_title="FRAME — AI Film Pre-Production Assistant",
                   page_icon=str(ICON) if ICON.is_file() else "🎬", layout="wide")
st.markdown(ui.CSS, unsafe_allow_html=True)  # "Cinematic Studio" visual layer (colours live in .streamlit/config.toml)

BUDGET_LABEL = {"LOW": "Бага", "MEDIUM": "Дунд", "HIGH": "Өндөр"}
STAGE_TITLE = {  # short names: used in the workflow stepper and notices
    Stage.IDEA: "Идея", Stage.SCRIPT: "Сценар", Stage.SCENES: "Дүр зураг",
    Stage.SHOTS: "Шот", Stage.PLAN: "Төлөвлөгөө", Stage.VALIDATION: "Шалгалт",
}


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
    """Badge that tells sample data, AI output and rule-based results apart."""
    origin = st.session_state.project.provenance.get(stage, "")
    if origin == "sample":
        html = ui.badge("Sample · статик жишээ — AI үр дүн биш", "sample")
    elif origin.startswith("deterministic+model:"):
        html = ui.badge(f"Rules + AI review · {origin.split(':', 1)[1]}", "ai")
    elif origin.startswith("model:"):
        html = ui.badge(f"AI · {origin.split(':', 1)[1]}", "ai")
    elif origin.startswith("deterministic"):
        html = ui.badge("Rules · deterministic", "rules")
    else:
        return
    st.markdown(html, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
def render_sidebar(p: Project) -> None:
    ss = st.session_state
    with st.sidebar:
        st.markdown(ui.side_label("Төсөл"), unsafe_allow_html=True)
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

        st.markdown(ui.side_label("AI загвар"), unsafe_allow_html=True)
        status = ss.llm_status
        cfg = ss.llm_config
        st.markdown(ui.chip("бэлэн" if status.available else "боломжгүй", "ok" if status.available else "bad")
                    + ui.chip(f"{cfg.provider} · {ss.provider.model or '—'}"), unsafe_allow_html=True)
        if not status.available:
            st.warning(status.message)
        with st.expander("Загварын тохиргоо"):
            if cfg.provider == "ollama":
                st.text_input("Ollama URL", value=cfg.base_url, key="w_base_url")
                st.text_input("Загварын нэр", value=cfg.model, key="w_model")
                st.number_input("Timeout (сек)", min_value=10, max_value=1800, value=int(cfg.timeout_seconds),
                                key="w_timeout")
                st.button("Хэрэглэх & дахин шалгах", on_click=apply_model_settings)
            else:
                st.caption(f"Провайдер: **{cfg.provider}**. Загвар, түлхүүрийг Streamlit Secrets эсвэл "
                           "`.env` (GEMINI_MODEL, GEMINI_API_KEY)-ээс уншина; түлхүүр энд харагдахгүй. "
                           "Тохиргоо өөрчилвөл апп-аа дахин ачаална уу.")
        if cfg.provider == "gemini":
            st.caption("⚠️ Gemini free tier-д илгээсэн агуулга Google-ийн бүтээгдэхүүнийг сайжруулахад "
                       "ашиглагдаж болно. Нууц агуулга бүү илгээгээрэй.")
        st.button("Холболтыг дахин шалгах", on_click=recheck_model)

        st.markdown(ui.side_label("Жишээ төсөл"), unsafe_allow_html=True)
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
# Workspace header: project slate, workflow stepper, content-origin notices
# --------------------------------------------------------------------------- #
def render_header(p: Project) -> None:
    ss = st.session_state
    status = ss.llm_status
    chips = (ui.chip("AI бэлэн" if status.available else "Demo горим · AI боломжгүй",
                     "ok" if status.available else "bad")
             + ui.chip(f"{ss.llm_config.provider} · {ss.provider.model or '—'}"))
    slate = [
        ("Жанр", p.genre),
        ("Хугацаа", f"{p.duration_minutes:g} мин"),
        ("Байршил", f"≤ {p.max_locations}" if p.max_locations else ""),
        ("Жүжигчин", f"≤ {p.max_actors}" if p.max_actors else ""),
        ("Төсөв", BUDGET_LABEL[p.budget.value]),
        ("Хэв маяг", p.visual_style),
    ]
    logo_col, info_col = st.columns([3, 5], vertical_alignment="center")
    with logo_col:
        if LOGO.is_file():  # the logo already carries the FRAME wordmark and tagline
            st.image(str(LOGO), width=330)
        else:
            st.markdown('<div class="frame-wordmark">FRAME</div>', unsafe_allow_html=True)
    with info_col:
        st.markdown(ui.header(p.name, slate, chips), unsafe_allow_html=True)
    st.markdown(ui.rule(), unsafe_allow_html=True)

    steps = [(STAGE_TITLE[s], p.get_status(s).value, p.provenance.get(s) == "sample") for s in Stage]
    st.markdown(ui.stepper(steps, sum(1 for s in Stage if p.is_fresh(s)), len(Stage),
                           NEXT_LABEL[next_intent(p)]), unsafe_allow_html=True)

    sample_stages = [STAGE_TITLE[s] for s in Stage if p.provenance.get(s) == "sample"]
    if sample_stages:
        st.markdown(ui.notice(
            f"<b>SAMPLE контент</b> · {ui.esc(', '.join(sample_stages))} нь урьдчилан бэлтгэсэн статик жишээ бөгөөд "
            "AI-ийн үр дүн <b>биш</b>. Өөрийн санаагаар AI-аар үүсгэхийн тулд «Санаа хөгжүүлэх» эсвэл "
            "«Сценар үүсгэх»-ийг ажиллуулна уу.", "sample"), unsafe_allow_html=True)
    if not status.available:
        st.warning("**Demo горим** — AI загвар холбогдоогүй. Дүрэмд суурилсан tool-ууд (шалгалт, төлөвлөгөө, "
                   "экспорт) ажиллана. Загвар шаардсан алхмууд тодорхой алдаа харуулна. "
                   f"({status.message})")


# --------------------------------------------------------------------------- #
# Chat column
# --------------------------------------------------------------------------- #
def render_chat() -> None:
    ss = st.session_state
    st.markdown(ui.panel_title("Туслах", "Director's assistant"), unsafe_allow_html=True)
    box = st.container(height=520, border=True)
    with box:
        if not ss.chat:
            with st.chat_message("assistant", avatar=":material/movie:"):
                st.markdown("Сайн байна уу! Би **FRAME** — таны киноны бэлтгэл ажлын туслах.\n\n"
                            "Киноны санаагаа бичээрэй, эсвэл зүүн талын **жишээ төсөл** сонгоно уу. "
                            "Би санаа → сценар → дүр зураг → шот → төлөвлөгөө → шалгалт → экспорт "
                            "дарааллаар хамт ажиллана.\n\n"
                            "Жишээ команд: `Санаагаа хөгжүүл` · `Create the full script package` · `Шалгалт хий`")
        for msg in ss.chat:
            avatar = ":material/person:" if msg["role"] == "user" else ":material/movie:"
            with st.chat_message(msg["role"], avatar=avatar):
                if msg.get("ok") is False:  # keep the full message visible, but make failures unmistakable
                    (st.error if "❌" in msg["content"] else st.warning)(msg["content"])
                else:
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
        '<div class="frame-hero"><div class="t">Шинэ төслөө эхлүүлье</div><ol>'
        "<li>Хажуугийн самбарт <b>жишээ төсөл</b> сонгох, эсвэл чатад киноны <b>санаагаа</b> бичих</li>"
        "<li>Туслах сценар, дүр зураг, шот, төлөвлөгөөг алхам алхмаар үүсгэнэ</li>"
        "<li>Шалгалт хийж, <b>Markdown багц</b> татаж авна</li></ol></div>",
        unsafe_allow_html=True)


def table(rows: list[dict], **column_config) -> None:
    """Read-only table with the shared look (and optional per-column widths)."""
    st.dataframe(rows, hide_index=True, width="stretch", column_config=column_config or None)


def tab_idea(p: Project) -> None:
    st.markdown(ui.section("Киноны санаа", "Санаагаа бичээд, батлагдсан шийдвэрүүдээ тэмдэглэнэ. "
                                          "AI эдгээрийг өөрчлөхгүй хадгална."), unsafe_allow_html=True)
    c1, c2 = st.columns([3, 2])
    c1.text_area("Киноны санаа (premise)", key="w_idea", height=140,
                 placeholder="Жишээ: Хоосон автобусны зогсоолд хүлээж буй охин ирээдүйг 10 секундээр харна…")
    c2.text_area("Батлагдсан бүтээлч шийдвэрүүд (мөр бүрт нэг) — AI эдгээрийг хадгална", key="w_decisions",
                 height=140)
    action_button("Санаа хөгжүүлэх", Intent.DEVELOP_IDEA, key="btn_idea", primary=True)

    notes = p.idea_notes
    if notes:
        st.markdown(ui.section("Хөгжүүлсэн санаа"), unsafe_allow_html=True)
        provenance_chip(Stage.IDEA)
        st.markdown(ui.notice(f"<b>Дүгнэлт.</b> {ui.esc(notes.summary)}"), unsafe_allow_html=True)
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
                with st.container(border=True):
                    c1, c2 = st.columns([5, 1], vertical_alignment="center")
                    c1.write(d)
                    c2.button("Батлах", key=f"approve_{i}", on_click=approve_decision, args=(d,),
                              disabled=d in p.approved_decisions)

    if p.logline:
        st.markdown(ui.section("Story", "Логлайн, синопсис, дүрүүд"), unsafe_allow_html=True)
        provenance_chip(Stage.SCRIPT)
        stale_banner(Stage.SCRIPT)
        st.markdown(f'<div class="frame-logline">{ui.esc(p.logline)}</div>', unsafe_allow_html=True)
        st.markdown(f"**Синопсис**\n\n{p.synopsis}")
        st.markdown("**Дүрүүд**")
        for start in range(0, len(p.characters), 3):
            for col, c in zip(st.columns(3), p.characters[start:start + 3]):
                with col, st.container(border=True):
                    st.markdown(ui.character_card(c.name, c.role, c.description, c.details),
                                unsafe_allow_html=True)


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

    st.markdown(ui.section("Сценар", "Дүр зураг бүрийн текст"), unsafe_allow_html=True)
    stale_banner(Stage.SCRIPT)
    if not p.script_scenes:
        st.info("Сценар хараахан үүсээгүй байна. Эхлээд санаагаа оруулаад «Сценар үүсгэх» дарна уу.")
    else:
        provenance_chip(Stage.SCRIPT)
        for s in p.script_scenes:
            with st.expander(f"Дүр зураг {s.number} — {s.heading}", expanded=s.number == 1):
                st.markdown(ui.screenplay(s.text), unsafe_allow_html=True)

    st.markdown(ui.section("Дүр зургийн задлал", "Байршил, цаг, дүрүүд, props, хугацаа"), unsafe_allow_html=True)
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
        table([{
            "#": s.number, "Байршил": s.location, "Цаг": s.time_of_day.value,
            "Дүрүүд": ", ".join(s.characters), "Props": ", ".join(s.props), "Үйлдэл": s.action,
            "Сек": s.estimated_duration_sec} for s in p.scenes],
            **{"#": st.column_config.NumberColumn("#", width="small"),
               "Цаг": st.column_config.TextColumn("Цаг", width="small"),
               "Сек": st.column_config.NumberColumn("Сек", format="%g", width="small"),
               "Үйлдэл": st.column_config.TextColumn("Үйлдэл", width="large")})


def tab_shots(p: Project) -> None:
    args = instruction_input("w_instr_shots")
    action_button("Шотын жагсаалт үүсгэх / дахин үүсгэх" if p.shots else "Шотын жагсаалт үүсгэх",
                  Intent.CREATE_SHOT_LIST, key="btn_shots", args=args, primary=not p.shots)
    stale_banner(Stage.SHOTS)
    if not p.shots:
        st.info("Шотын жагсаалт хараахан үүсээгүй байна. Эхлээд дүр зургийг задлах шаардлагатай.")
        return
    st.markdown(ui.section("Шотын жагсаалт", "Дүр зураг бүрт холбогдсон камерын шотууд"), unsafe_allow_html=True)
    provenance_chip(Stage.SHOTS)
    m1, m2 = st.columns(2)
    m1.metric("Шот", len(p.shots))
    m2.metric("Нийт хугацаа", f"{sum(s.estimated_duration_sec for s in p.shots):g} сек")
    table([{
        "#": s.number, "Scene": s.scene_number, "Тайлбар": s.description, "Type": s.shot_type.value,
        "Angle": s.camera_angle.value, "Movement": s.movement.value, "Дуу": s.audio_notes,
        "Сек": s.estimated_duration_sec} for s in p.shots],
        **{"#": st.column_config.NumberColumn("#", width="small"),
           "Scene": st.column_config.NumberColumn("Scene", width="small"),
           "Тайлбар": st.column_config.TextColumn("Тайлбар", width="large"),
           "Type": st.column_config.TextColumn("Type", width="medium"),
           "Дуу": st.column_config.TextColumn("Дуу", width="medium"),
           "Сек": st.column_config.NumberColumn("Сек", format="%g", width="small")})


def tab_plan(p: Project) -> None:
    action_button("Төлөвлөгөө гаргах / шинэчлэх", Intent.PLAN_PRODUCTION, key="btn_plan", primary=not p.plan)
    stale_banner(Stage.PLAN)
    if not p.plan:
        st.info("Төлөвлөгөө хараахан гараагүй байна. Шотын жагсаалт бэлэн болсны дараа ажиллана.")
        return
    plan = p.plan
    st.markdown(ui.section("Үйлдвэрлэлийн төлөвлөгөө"), unsafe_allow_html=True)
    provenance_chip(Stage.PLAN)
    st.caption("Доорх зүйлс нь дүрэмд суурилсан ТААМАГЛАЛ. Үнэ, зардал тооцоогүй.")
    m1, m2, m3 = st.columns(3)
    m1.metric("Авалтын өдөр", plan.shoot_days)
    m2.metric("Нийт (таамаг)", f"~{plan.total_minutes} мин")
    m3.metric("Props", len(plan.props))
    st.markdown(ui.section("Хуваарь"), unsafe_allow_html=True)
    table([{"Өдөр": b.day, "Дараалал": b.order, "Блок": b.label,
            "Дүр зураг": ", ".join(map(str, b.scene_numbers)), "Мин (таамаг)": b.estimated_minutes,
            "Тэмдэглэл": b.notes} for b in plan.schedule],
          **{"Өдөр": st.column_config.NumberColumn("Өдөр", width="small"),
             "Дараалал": st.column_config.NumberColumn("Дараалал", width="small")})
    st.markdown(ui.section("Тоног төхөөрөмж"), unsafe_allow_html=True)
    mark = {True: "✅ байгаа", False: "❓ жагсаалтад алга", None: "—"}
    table([{"Нэр": e.name, "Шалтгаан": e.reason, "Төлөв": mark[e.available]} for e in plan.equipment],
          **{"Шалтгаан": st.column_config.TextColumn("Шалтгаан", width="large")})
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(ui.section("Props"), unsafe_allow_html=True)
        table([{"Нэр": x.name, "Дүр зураг": ", ".join(map(str, x.scene_numbers))} for x in plan.props])
    with c2:
        st.markdown(ui.section("Байршил"), unsafe_allow_html=True)
        table([{"Байршил": x.name, "Дүр зураг": ", ".join(map(str, x.scene_numbers)), "Цаг": x.time_of_day}
               for x in plan.locations])
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
        st.markdown(ui.section("Шалгалтын дүн", "Дүрмэн шалгалт баталгаатай; AI санал нь зөвхөн зөвлөмж"),
                    unsafe_allow_html=True)
        provenance_chip(Stage.VALIDATION)
        counts = {sev: sum(1 for i in p.issues if i.severity == sev) for sev in Severity}
        m1, m2, m3 = st.columns(3)
        m1.metric("Алдаа", counts[Severity.ERROR])
        m2.metric("Анхааруулга", counts[Severity.WARNING])
        m3.metric("Тэмдэглэл", counts[Severity.INFO])
        if not p.issues:
            st.success("Дүрмийн шалгалтаар асуудал илрээгүй.")
        for issue in sorted(p.issues, key=lambda i: list(Severity).index(i.severity)):
            where = " · ".join(filter(None, [
                f"дүр зураг {issue.scene_number}" if issue.scene_number else "",
                f"шот {issue.shot_number}" if issue.shot_number else ""]))
            source = "дүрмэн шалгалт" if issue.source == "deterministic" else "AI санал · баталгаагүй"
            st.markdown(ui.issue_card(issue.severity.value, issue.code, issue.message, issue.suggestion,
                                      source, where), unsafe_allow_html=True)

    st.divider()
    st.markdown(ui.section("Экспорт", "Бүх хэсэг, шалгалтын тэмдэглэлийг агуулсан Markdown багц"),
                unsafe_allow_html=True)
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
    render_header(p)

    chat_col, work_col = st.columns([4, 8], gap="large")
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
