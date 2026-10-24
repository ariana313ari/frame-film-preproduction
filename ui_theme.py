"""Visual layer for the FRAME "Cinematic Studio" UI.

Presentation only: CSS plus small HTML builders. No application state or business logic.

* Colours/radii/borders are mostly set natively in `.streamlit/config.toml`; the CSS here covers
  what the theme cannot (header, stepper, badges, screenplay page) plus light polish of a few
  Streamlit elements via their stable `data-testid` hooks.
* Every dynamic value goes through `esc()`: some of it (validation messages, model names) can come
  from a model or the environment and must never be interpreted as HTML.
* HTML strings are built on ONE line without indentation: Markdown treats indented lines as code.
"""
from __future__ import annotations

from html import escape
from typing import Iterable

# Design tokens (kept in sync with .streamlit/config.toml)
TOKENS = {
    "bg": "#0d0e11", "surface": "#15171c", "surface2": "#1b1e25", "line": "#2a2d36", "line2": "#3a3e4a",
    "text": "#e9e5db", "muted": "#9fa1a8", "gold": "#c9a45c", "gold_hi": "#d9b872", "on_gold": "#14110a",
    "red": "#e07a72", "orange": "#d9904f", "blue": "#7aa2d4", "green": "#82b896", "violet": "#a58fd0",
}

CSS = """
<style>
:root {
  --f-bg:#0d0e11; --f-surface:#15171c; --f-surface2:#1b1e25; --f-line:#2a2d36; --f-line2:#3a3e4a;
  --f-text:#e9e5db; --f-muted:#9fa1a8; --f-gold:#c9a45c; --f-gold-hi:#d9b872; --f-on-gold:#14110a;
  --f-red:#e07a72; --f-orange:#d9904f; --f-blue:#7aa2d4; --f-green:#82b896; --f-violet:#a58fd0;
  --f-serif: "Source Serif", "Source Serif 4", Georgia, "Times New Roman", serif;
  --f-sans: "Source Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
}
.block-container { padding-top: 3.6rem; padding-bottom: 3rem; max-width: 1600px; }
[data-testid="stAppDeployButton"] { display: none; }

/* ---- header ---------------------------------------------------------------------------- */
.frame-header { display:flex; flex-wrap:wrap; align-items:center; justify-content:space-between; gap:.8rem 2rem; }
/* The logo PNG has an opaque near-black background a few levels off the page colour: blend it away. */
[data-testid="stImage"] img { mix-blend-mode: lighten; }
.frame-rule { margin:.5rem 0 .9rem; border-bottom:1px solid var(--f-line); }
.frame-wordmark { font-family:var(--f-serif); font-size:1.6rem; letter-spacing:.34em; color:var(--f-gold); line-height:1; }
.frame-wordmark small { display:block; margin-top:.4rem; font-family:var(--f-sans); font-size:.62rem; letter-spacing:.22em;
  color:var(--f-muted); text-transform:uppercase; }
.frame-project { flex:1 1 22rem; min-width:0; }
.frame-project-title { font-family:var(--f-serif); font-size:1.4rem; color:var(--f-text); line-height:1.2;
  overflow-wrap:anywhere; }
.frame-slate { display:flex; flex-wrap:wrap; gap:.2rem 1.2rem; margin-top:.35rem; font-size:.8rem; color:var(--f-muted); }
.frame-slate b { margin-right:.4rem; font-size:.64rem; font-weight:600; letter-spacing:.1em; text-transform:uppercase;
  color:var(--f-gold); }
.frame-status { display:flex; flex-wrap:wrap; gap:.4rem; align-items:center; }

/* ---- chips & badges --------------------------------------------------------------------- */
.frame-chip { display:inline-block; padding:.12rem .6rem; margin:0 .3rem .2rem 0; border:1px solid var(--f-line2);
  border-radius:999px; font-size:.76rem; line-height:1.4; color:var(--f-muted); white-space:nowrap; }
.frame-chip.ok { color:var(--f-green); border-color:#3c5e49; }
.frame-chip.bad { color:var(--f-red); border-color:#74403c; }
.frame-badge { display:inline-block; margin:.1rem 0 .5rem; padding:.14rem .55rem; border-radius:4px; font-size:.7rem;
  font-weight:600; letter-spacing:.03em; border:1px solid var(--f-line2); color:var(--f-muted); }
.frame-badge.ai { color:var(--f-gold-hi); border-color:#6b5a30; background:rgba(201,164,92,.08); }
.frame-badge.rules { color:var(--f-blue); border-color:#3d5878; background:rgba(122,162,212,.07); }
.frame-badge.sample { color:var(--f-violet); border:1px dashed #6f5ea3; background:rgba(165,143,208,.09); }

/* ---- workflow stepper ---------------------------------------------------------------------- */
.frame-stepper { display:flex; flex-wrap:wrap; gap:.4rem; margin:0 0 .45rem; padding:0; list-style:none; }
.frame-stepper li { display:flex; align-items:center; gap:.5rem; padding:.28rem .75rem .28rem .35rem;
  border:1px solid var(--f-line); border-radius:999px; background:var(--f-surface); color:var(--f-muted); font-size:.8rem; }
.frame-stepper .dot { display:inline-flex; align-items:center; justify-content:center; width:1.2rem; height:1.2rem;
  border:1px solid var(--f-line2); border-radius:50%; font-size:.68rem; }
.frame-stepper li.done { color:var(--f-text); border-color:#5d4e2c; }
.frame-stepper li.done .dot { background:var(--f-gold); border-color:var(--f-gold); color:var(--f-on-gold); font-weight:700; }
.frame-stepper li.outdated { color:var(--f-orange); border-color:#7a5530; }
.frame-stepper li.outdated .dot { border-color:var(--f-orange); }
.frame-stepper .tag { font-size:.62rem; letter-spacing:.06em; text-transform:uppercase; color:var(--f-violet); }
.frame-count { margin-left:.4rem; align-self:center; font-size:.78rem; color:var(--f-muted); }
.frame-next { margin:0 0 1rem; font-size:.82rem; color:var(--f-muted); }
.frame-next b { color:var(--f-gold-hi); font-weight:600; }
.sr { position:absolute; left:-9999px; }

/* ---- sections, notices, cards ----------------------------------------------------------------- */
.frame-section { margin:1.1rem 0 .5rem; }
.frame-section .t { font-family:var(--f-serif); font-size:1.2rem; color:var(--f-text); }
.frame-section .s { margin-top:.1rem; font-size:.84rem; color:var(--f-muted); }
.frame-notice { margin:.2rem 0 .9rem; padding:.65rem .95rem; border:1px solid var(--f-line); border-left:3px solid var(--f-line2);
  border-radius:6px; background:var(--f-surface); font-size:.86rem; line-height:1.5; }
.frame-notice.sample { border:1px dashed #6f5ea3; border-left:3px solid var(--f-violet);
  background:repeating-linear-gradient(135deg, rgba(165,143,208,.07) 0 9px, transparent 9px 18px); }
.frame-notice b { color:var(--f-text); }
.frame-hero { padding:1.1rem 1.3rem; border:1px solid var(--f-line); border-radius:8px; background:var(--f-surface); margin-bottom:1rem; }
.frame-hero .t { font-family:var(--f-serif); font-size:1.35rem; color:var(--f-gold-hi); margin-bottom:.5rem; }
.frame-hero ol { margin:0; padding-left:1.2rem; color:var(--f-text); line-height:1.8; }
.frame-logline { margin:.3rem 0 1rem; padding:.2rem 0 .2rem 1rem; border-left:3px solid var(--f-gold);
  font-family:var(--f-serif); font-size:1.3rem; line-height:1.5; color:var(--f-text); }
.frame-char .n { font-family:var(--f-serif); font-size:1.1rem; color:var(--f-text); }
.frame-char .r { margin:.1rem 0 .4rem; font-size:.68rem; letter-spacing:.1em; text-transform:uppercase; color:var(--f-gold); }
.frame-char .d { font-size:.86rem; line-height:1.5; color:var(--f-text); }
.frame-char .c { margin-top:.5rem; }
.frame-screenplay { padding:1rem 1.25rem; border:1px solid var(--f-line); border-radius:6px; background:#101115;
  font-family:"Courier New", Courier, monospace; font-size:.9rem; line-height:1.65; white-space:pre-wrap; color:#dedacd; }

/* ---- validation issue cards ------------------------------------------------------------------- */
.frame-issue { margin:.5rem 0; padding:.7rem .95rem; border:1px solid var(--f-line); border-left:4px solid var(--f-line2);
  border-radius:6px; background:var(--f-surface); }
.frame-issue.sev-error { border-left-color:var(--f-red); }
.frame-issue.sev-warning { border-left-color:var(--f-orange); }
.frame-issue.sev-info { border-left-color:var(--f-blue); }
.frame-issue .head { display:flex; flex-wrap:wrap; align-items:center; gap:.2rem .6rem; font-size:.78rem; color:var(--f-muted); }
.frame-issue .sev { font-weight:700; letter-spacing:.06em; }
.frame-issue.sev-error .sev { color:var(--f-red); } .frame-issue.sev-warning .sev { color:var(--f-orange); }
.frame-issue.sev-info .sev { color:var(--f-blue); }
.frame-issue code { font-size:.76rem; }
.frame-issue .msg { margin:.4rem 0 .15rem; color:var(--f-text); line-height:1.5; }
.frame-issue .fix { font-size:.84rem; color:var(--f-muted); }
.frame-issue .src { margin-left:auto; font-size:.72rem; font-style:italic; }

/* ---- light polish of Streamlit elements (stable data-testid hooks only) ----------------------- */
[data-testid="stBaseButton-primary"] { color:var(--f-on-gold); font-weight:600; }
[data-testid="stBaseButton-primary"] p { color:var(--f-on-gold); }
[data-testid="stBaseButton-secondary"] { background:transparent; }
[data-testid="stBaseButton-secondary"]:hover { border-color:var(--f-gold); color:var(--f-gold-hi); }
[data-testid="stMetric"] { padding:.7rem .95rem; border:1px solid var(--f-line); border-radius:6px; background:var(--f-surface); }
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * { white-space:normal !important; overflow:visible !important;
  text-overflow:clip !important; }
[data-testid="stMetricLabel"] { color:var(--f-muted); text-transform:uppercase; letter-spacing:.06em; font-size:.72rem; }
[data-testid="stAlert"] { border-radius:6px; }
[data-testid="stExpander"] { border-radius:6px; }
[data-testid="stTab"] { padding-left:.9rem; padding-right:.9rem; }
.frame-side-label { margin:1.2rem 0 .4rem; font-size:.68rem; font-weight:600; letter-spacing:.14em; text-transform:uppercase;
  color:var(--f-gold); }
.frame-panel-title { margin:0 0 .5rem; font-family:var(--f-serif); font-size:1.2rem; color:var(--f-text); }
.frame-panel-title small { margin-left:.5rem; font-family:var(--f-sans); font-size:.72rem; color:var(--f-muted); }

@media (max-width: 900px) {
  .block-container { padding: 3.6rem .8rem 2rem; }  /* keep clear of Streamlit's fixed top bar */
  .frame-wordmark { font-size:1.25rem; } .frame-project-title { font-size:1.15rem; }
  .frame-logline { font-size:1.1rem; }
}
</style>
"""


def esc(value: object) -> str:
    return escape(str(value), quote=True)


def chip(text: str, kind: str = "") -> str:
    return f'<span class="frame-chip {esc(kind)}">{esc(text)}</span>'


def badge(text: str, kind: str = "") -> str:
    return f'<span class="frame-badge {esc(kind)}">{esc(text)}</span>'


def section(title: str, subtitle: str = "") -> str:
    sub = f'<div class="s">{esc(subtitle)}</div>' if subtitle else ""
    return f'<div class="frame-section"><div class="t">{esc(title)}</div>{sub}</div>'


def side_label(text: str) -> str:
    return f'<div class="frame-side-label">{esc(text)}</div>'


def panel_title(title: str, hint: str = "") -> str:
    small = f"<small>{esc(hint)}</small>" if hint else ""
    return f'<div class="frame-panel-title">{esc(title)}{small}</div>'


def notice(text_html: str, kind: str = "") -> str:
    """`text_html` must already be safe HTML (built from esc()'d parts)."""
    return f'<div class="frame-notice {esc(kind)}">{text_html}</div>'


def rule() -> str:
    return '<div class="frame-rule"></div>'


def header(name: str, slate: Iterable[tuple[str, str]], chips_html: str) -> str:
    items = "".join(f"<span><b>{esc(k)}</b>{esc(v)}</span>" for k, v in slate if str(v).strip())
    return (
        '<div class="frame-header">'
        '<h1 class="sr">FRAME — AI Film Pre-Production Assistant</h1>'
        f'<div class="frame-project"><div class="frame-project-title">{esc(name)}</div>'
        f'<div class="frame-slate">{items}</div></div>'
        f'<div class="frame-status">{chips_html}</div></div>'
    )


def stepper(steps: Iterable[tuple[str, str, bool]], done: int, total: int, next_label: str) -> str:
    """steps: (title, state in {fresh, outdated, empty}, is_sample)."""
    marks = {"fresh": ("done", "✓", "бэлэн"), "outdated": ("outdated", "!", "хуучирсан"),
             "empty": ("", "", "хийгдээгүй")}
    items = []
    for title, state, is_sample in steps:
        css, mark, word = marks[state]
        tag = '<span class="tag">sample</span>' if is_sample else ""
        items.append(f'<li class="{css}"><span class="dot" aria-hidden="true">{mark}</span>'
                     f'<span>{esc(title)}</span><span class="sr">({word})</span>{tag}</li>')
    return (
        f'<ol class="frame-stepper" aria-label="Ажлын явц">{"".join(items)}'
        f'<li style="border:none;background:none" class="frame-count">{done}/{total}</li></ol>'
        f'<div class="frame-next">Дараагийн алхам: <b>{esc(next_label)}</b></div>'
    )


def character_card(name: str, role: str, description: str, details: Iterable[str]) -> str:
    chips = "".join(chip(d) for d in details)
    return (
        f'<div class="frame-char"><div class="n">{esc(name)}</div>'
        f'<div class="r">{esc(role) if role else "&nbsp;"}</div>'
        f'<div class="d">{esc(description)}</div>'
        f'<div class="c">{chips}</div></div>'
    )


def screenplay(text: str) -> str:
    return f'<div class="frame-screenplay">{esc(text)}</div>'


def issue_card(severity: str, code: str, message: str, suggestion: str, source_label: str, where: str) -> str:
    fix = f'<div class="fix">Санал: {esc(suggestion)}</div>' if suggestion else ""
    place = f"<span>{esc(where)}</span>" if where else ""
    return (
        f'<div class="frame-issue sev-{esc(severity.lower())}">'
        f'<div class="head"><span class="sev">{esc(severity)}</span><code>{esc(code)}</code>{place}'
        f'<span class="src">{esc(source_label)}</span></div>'
        f'<div class="msg">{esc(message)}</div>{fix}</div>'
    )
