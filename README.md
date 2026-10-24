# FRAME — AI Film Pre-Production Agent

FRAME turns a film idea into a practical pre-production package: **idea → logline & synopsis → script →
scene breakdown → shot list → production plan → continuity validation → Markdown export.**
It is built for beginner and student filmmakers, with a Mongolian interface and Mongolian output
(English technical terms such as *INT./EXT.*, *Close-up* are kept).

It is a small but real **agent**: an orchestrator receives your message and the project state, decides what is
needed, asks for missing information, verifies and executes registered Python tools, validates their output,
updates the project only with valid results, and then explains what happened and what to do next.

---

## Quick start

Requires **Python 3.11+**. No paid API, no GPU, no database, no Docker.

```bash
# 1. (optional) virtual environment
python -m venv .venv
.venv\Scripts\activate            # Windows      (macOS/Linux: source .venv/bin/activate)

# 2. install
pip install -r requirements.txt

# 3. install Ollama (https://ollama.com) and pull a model
ollama pull qwen2.5:3b

# 4. run
streamlit run app.py
```

Open the URL Streamlit prints (default <http://localhost:8501>).

**Try it in 30 seconds without a model:** in the sidebar click **“The Last Bus — бүрэн demo (static)”**,
then open **Validation** → *Шалгалт ажиллуулах* → *Үйлдвэрлэлийн багц бэлтгэх* → *Markdown татаж авах*.
The *“Demo: алдаатай жишээ”* button loads the same project with deliberate mistakes so you can see validation findings.

### Run the tests
```bash
python -m pytest
```
Tests need no internet, no Ollama and no API key: a scripted LLM stub and a fake Gemini client exist **only** in `tests/`,
and the suite ignores your real `.env` and `FRAME_*`/`GEMINI_*` variables so it can never use your key.

---

## Configuration

Copy `.env.example` to `.env` (optional). Everything also works with defaults.

| Variable | Default | Meaning |
|---|---|---|
| `FRAME_LLM_PROVIDER` | `ollama` | `ollama` or `gemini`. No automatic fallback between them. |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server |
| `FRAME_MODEL` | `qwen2.5:3b` | model name |
| `FRAME_TIMEOUT_SECONDS` | `300` | per-request timeout (CPU generation is slow) |
| `FRAME_TEMPERATURE` | `0.4` | sampling temperature |
| `FRAME_NUM_CTX` | `8192` | context window requested from Ollama |
| `GEMINI_API_KEY` | – | Gemini only. Secret: put it in your local `.env` only. |
| `GEMINI_MODEL` | – (`.env.example` suggests one) | Gemini only. Required; model names and free-tier access change. |
| `GEMINI_TEMPERATURE` | unset | Gemini only, optional (see below). |

With Ollama the URL, model name and timeout can also be changed live in the sidebar (*Загварын тохиргоо*).
The default Ollama setup needs no API key. No key is hard-coded, and the UI never shows secrets.
`.env` is read when the app starts; restart Streamlit after editing it. A variable that is already set (non-empty)
in your shell wins over `.env`. `.env` is ignored by Git — never commit it.

### Using Google Gemini (optional)
Uses the official [`google-genai`](https://pypi.org/project/google-genai/) SDK (`models.generate_content`).

1. Create an API key in [Google AI Studio](https://aistudio.google.com/) and put it in your local `.env`
   (never paste it into the app, chat or an issue):
   ```
   FRAME_LLM_PROVIDER=gemini
   GEMINI_API_KEY=<your key>
   GEMINI_MODEL=<model name>
   ```
2. Pick `GEMINI_MODEL` from [Google's model list](https://ai.google.dev/gemini-api/docs/models) and check free-tier
   access on the [pricing page](https://ai.google.dev/gemini-api/docs/pricing). FRAME does not hard-code a model.
3. `pip install -r requirements.txt`, then `streamlit run app.py`. The sidebar shows whether the configuration is
   *incomplete* (missing key/model) or the key and model are *confirmed* by the API.

Things to know:
* **Privacy:** on the free tier Google may use submitted content to improve its products. Do not send confidential
  scripts. The app shows a reminder when Gemini is active.
* **Quota:** free-tier rate limits are not published as fixed numbers; see your limits in AI Studio. One chat turn can
  make several requests (up to 4 tools, plus repairs). A `429` is reported as a rate-limit error — wait and retry.
* **Temperature:** Google recommends the default temperature for Gemini 3 models (lower values can cause looping), so
  FRAME sends none unless you set `GEMINI_TEMPERATURE`; per-call temperatures from the agent are ignored for Gemini.
* **Status check:** the sidebar check calls `models.get` (no generation). It confirms the key and model exist; it cannot
  prove you have free-tier quota for generation.
* The key is read from the environment on every request and is never stored in the config, the provider object, or
  Streamlit session state, and error messages are scrubbed of it.

### Choosing a model
`qwen2.5:3b` is the default because it runs on a CPU-only laptop. **Its Mongolian is limited** and small models
sometimes produce invalid JSON. FRAME retries once with the validation error and then shows an honest error — it
never fabricates structured data. If your hardware allows, try a larger model (`qwen2.5:7b`, `llama3.1:8b`, …)
by setting `FRAME_MODEL`. Availability depends on what Ollama currently offers; check <https://ollama.com/library>.
Expect each generation step to take from tens of seconds to several minutes on CPU.

---

## How the agent works

```
user message / button
   │
   ▼
analyse project ──► decide intent ──► VERIFY (tool exists? arguments valid?) ──► missing info? ask
                    │ button / keywords / model proposal                       prerequisites fresh? else suggest
                    ▼
        execute registered tool (max 4 per turn) ──► validate output (Pydantic + rules)
                    ▼
        update Project only if valid ──► explain + offer next step
```

* **Intents:** `DEVELOP_IDEA`, `BUILD_SCRIPT`, `BREAKDOWN_SCENES`, `CREATE_SHOT_LIST`, `PLAN_PRODUCTION`,
  `CHECK_CONTINUITY`, `EXPORT_PACKAGE`, `GENERAL_QUESTION`.
* **Routing:** UI buttons carry an explicit intent; clear Mongolian/English keywords are routed locally (fast);
  anything else is *proposed* by the model as JSON. The orchestrator rejects unknown tools and invalid arguments
  (`extra="forbid"` Pydantic models) before anything runs.
* **Bounded:** at most **4 tool executions per turn** (`agent.MAX_TOOL_CALLS`), no recursion. A request such as
  “Create the full script package” runs script → scenes → shots → plan, then stops and offers *validation* next.
* **Invalid output:** one controlled repair (the validation error is fed back to the model), then a clear error.
* **No hidden reasoning is shown** — only short status lines (“Сценар бичиж байна”, …) and the list of tools executed.
* **Honest reports:** step summaries are produced by Python from validated data, not by the model.

### Single source of truth and staleness
`st.session_state.project` (a Pydantic `Project`) holds all data; the chat transcript is only text.
Each stage has a status (`empty` / `fresh` / `outdated`). Regenerating the **script** marks scenes, shots, plan and
validation `outdated`; regenerating **scenes** marks shots, plan and validation `outdated`; regenerating **shots**
marks plan and validation `outdated`. Outdated artifacts stay visible with a warning, tools refuse to build on them,
and the export flags them. *Approved decisions* (Idea & Story tab) are passed to the model with a “MUST be preserved”
instruction; a change request goes in the optional *Нэмэлт заавар* field or in the chat message.

### Tools (`tools.py`)
| Tool | Type | Notes |
|---|---|---|
| `develop_idea` | LLM | logline options, themes, suggested decisions, questions |
| `build_script` | LLM | logline, synopsis, characters, screenplay scenes |
| `breakdown_scenes` | LLM | location, time of day, characters, props, action, duration |
| `create_shot_list` | LLM | shot type, angle, movement, description, audio, duration; scene links verified |
| `check_continuity` | **deterministic** + optional LLM | rule findings are `deterministic`; AI suggestions are labelled `model` and capped at WARNING |
| `estimate_resources` | **deterministic** | props, equipment, locations, schedule from stated rule-of-thumb assumptions; **no prices** |
| `export_package` | deterministic | Markdown document with all sections, provenance and validation notes |

Deterministic checks: unknown/unused characters, scene & shot numbering, shot→scene references, positive durations,
film duration vs. target (warning >15 %, error >40 %), shots vs. scene duration, location and actor limits,
equipment vs. camera movement, prop gaps (heuristic text matching, labelled as such).
Semantic continuity (e.g. costume changes) is only a model suggestion and is **not** guaranteed.

### Demo mode (no model)
If Ollama or the model is unavailable the sidebar shows it, a banner appears, and LLM-dependent actions fail with a
clear message while `check_continuity`, `estimate_resources` and `export_package` keep working. The sample buttons load
**explicitly labelled static sample data** (`provenance = sample`); it is never presented as model output.

---

## Demo script: “The Last Bus”
Fictional creative prompt — mystery thriller, 3 min, 1 location, 2 actors, low budget, rainy-night suspense.

1. Sidebar → **The Last Bus — зөвхөн brief**.
2. Chat: `Санаагаа хөгжүүл` → Idea & Story tab shows logline options; click **Батлах** on a suggested decision.
3. Chat: `Create the full script package` → script, scenes, shot list and production plan (4 tools, then it stops).
4. **Validation** tab → *Шалгалт ажиллуулах* (optionally tick the AI review).
5. *Үйлдвэрлэлийн багц бэлтгэх* → *Markdown татаж авах*.

Without a model, use **бүрэн demo (static)** or **алдаатай жишээ** and start at step 4.

---

## Project layout
```
app.py            Streamlit UI (sidebar, chat, 5 result tabs)
agent.py          orchestrator: routing, verification, bounded execution, explanations
tools.py          the tools, deterministic rules, tool registry
schemas.py        Pydantic models (Project, Character, Scene, Shot, ProductionPlan, ValidationIssue, ToolResult, …)
llm_provider.py   provider interface, Ollama + Gemini adapters, JSON extraction + one-shot repair
prompts.py        all model prompts
export_utils.py   Markdown export
sample_project.py "The Last Bus" sample data (labelled as sample)
tests/            pytest suite (scripted LLM stub and fake Gemini client; hermetic, no network)
```

### Adding a provider
Subclass `LLMProvider` (`check_status()` and `chat(messages, json_mode, temperature)`), raise the typed `LLMError`
subclasses on failure, register the class in `llm_provider.PROVIDERS`, and set `FRAME_LLM_PROVIDER`. Read any API key
from an environment variable — never from code or the UI. Ollama and Gemini ship today; `GeminiProvider` is a good template
(it also shows how to inject a fake client for offline tests).

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| Sidebar says *боломжгүй* / “Ollama руу холбогдож чадсангүй” | Install Ollama, run `ollama serve` (or start the desktop app), check `OLLAMA_BASE_URL`, click *Холболтыг дахин шалгах*. |
| “загвар суугаагүй” / “олдсонгүй” | `ollama pull qwen2.5:3b` or set `FRAME_MODEL` to an installed model (`ollama list`). |
| Gemini: “GEMINI_API_KEY тохируулагдаагүй” / “Тохиргоо дутуу” | Set `GEMINI_API_KEY` and `GEMINI_MODEL` in `.env` (or the shell), then restart Streamlit. |
| Gemini: “түлхүүр буруу эсвэл эрхгүй” (401/403) | Check the key; the project may not be eligible for the free tier. |
| Gemini: “загвар олдсонгүй” (404) | `GEMINI_MODEL` is wrong, retired, or not available to your account — pick one from the models page. |
| Gemini: “хязгаар/квот хэтэрлээ” (429) | Free-tier limit reached. Wait, or reduce requests per turn; check limits in AI Studio. |
| Timeout | CPU generation is slow. Raise `FRAME_TIMEOUT_SECONDS`, use a smaller model, or shorten the film duration (fewer scenes). |
| “Загварын гаргасан JSON буруу” | Small models sometimes break JSON. Click the button again, add a clearer premise, or use a larger model. Nothing is saved when output is invalid. |
| Mongolian text is poor | Use a larger/multilingual model, or switch *Үр дүнгийн хэл* to English. |
| Cyrillic characters garbled in a Windows terminal | Set `PYTHONUTF8=1` (the app itself is unaffected). |
| `ModuleNotFoundError` | Activate the virtual environment and run `pip install -r requirements.txt`. |
| Port already in use | `streamlit run app.py --server.port 8502`. |

---

## Known limitations
* The Gemini adapter was verified against a fake client and the real SDK's request validation (loopback only); it has
  **not** been run against the live Gemini API in the development environment, so check it with your own key.
* The pipeline was verified with the scripted test LLM and the Streamlit headless test runner; **no real Ollama model
  was available in the development environment**, so output quality and speed with `qwen2.5:3b` (especially in
  Mongolian) are untested and should be checked on your machine.
* Generated content is read-only in the UI (regenerate with instructions rather than hand-edit); the project can be
  saved as JSON from the sidebar but there is no import button.
* Rule-based schedule/resource estimates are rough heuristics, not production guarantees; no costs are produced.
* Semantic continuity review depends on the model and is advisory only.
* Session state lives in the browser session; closing the tab loses unsaved work (export or save the JSON).
