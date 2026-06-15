# ODNN Streamlit Frontend Development Guide

## Workflow & Behavior: Explore Before Execution (CRITICAL)
- **Step 1: Explore (先探索):** Before modifying or creating ANY file, you MUST explore the codebase. Use tools to read file contents, search for existing variables, and understand the current state of adapters and Streamlit session states. **NEVER guess file paths or internal code structures.**
- **Step 2: Plan (再计划):** After exploring, present a concise plan detailing exactly which files will be modified and what logic will be added. 
- **Step 3: Execute (后执行):** Only write or modify code after the context is fully verified and the plan is established. Stop and ask the user for clarification if you find conflicting logic.

## Commands
- Run application: `streamlit run frontend/app.py`
- Run training standalone: `python mainfor6.py --config <path.json> --output_dir <dir>`
- Lint & Type check: `mypy frontend/`

## Core Architecture: Adapter Pattern
- **Strict Separation:** Frontend code strictly lives in `frontend/`. Never directly import or execute heavy training logic (e.g., `mainfor6.py`) into UI files.
- **Adapters:** UI must communicate with training scripts via classes in `frontend/adapters/` (inheriting from `BaseODNNAdapter`).
- **Subprocess Training:** Training must run as a detached subprocess. The frontend monitors progress by tailing log files (`metrics.jsonl`, `training.log`).

## Streamlit State & Performance (CRITICAL)
- **No Blocking Loops:** NEVER use `while True` + `time.sleep()`. Use `streamlit-autorefresh` for real-time log polling and UI updates.
- **Caching:** Wrap heavy I/O operations (reading `.mat` or `.npz` files) with `@st.cache_data`.
- **Session State:** Store configuration dicts, adapter selections, and widget states in `st.session_state` to prevent data loss across reruns.
- **Config Contract:** Always pipe configuration loads through `ConfigManager.merge_with_defaults()` to ensure backward compatibility when new hyperparameters are added.

## UI & Coding Conventions
- **Language:** ALL user-facing UI text MUST be in English. Zero Chinese characters in the UI.
- **Emojis:** NEVER use emojis in code, UI, or comments.
- **Comments:** Comments can be a mix of English and Chinese. Keep them pragmatic and human-written in tone; explain the "why" behind complex state management or tensor shapes.
- **Naming:** 
  - `snake_case` for files, functions, variables (e.g., `mainfor6_adapter.py`, `load_config`).
  - `PascalCase` for classes (e.g., `Mainfor6Adapter`).
  - `UPPER_CASE` for constants.
- **Type Hints:** Enforce strong typing (`typing.Dict`, `typing.List`, `np.ndarray`), especially in Adapter interfaces and data parsing functions.

## Error Handling
- **Defensive UI:** Handle missing config files gracefully by falling back to defaults.
- **Crash Detection:** Do not rely solely on process exit codes. Read the tail of `training.log` to detect "Error", "Traceback", or "Exception" keywords and display them using `st.error` inside an `st.expander`.