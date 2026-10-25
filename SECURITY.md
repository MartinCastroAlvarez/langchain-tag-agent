# Security

langchain-tag-agent lets a language model write SQL against a database and read its rows.
People install it and point it at their own databases and models. Design goals: model
text can't change, read or exfiltrate what it shouldn't, rows can't take over the model's
instructions, nothing leaves the machine unless you opt in, and no question runs without
bound.

## Reporting a vulnerability

Open a private security advisory at
<https://github.com/MartinCastroAlvarez/langchain-tag-agent/security/advisories/new>.
Do **not** open a public issue for a suspected vulnerability. We acknowledge within 5
business days and aim to ship a fix within 30 days. You are credited in the advisory
unless you ask otherwise.

Supported versions: the latest minor release.

## 1. Threat model

**Assets.** The database file and every column you did not allowlist, the host and its
filesystem, the model endpoint (and its key, if you use a hosted one).

| Actor | Controls | Must not be able to |
| --- | --- | --- |
| Whoever asks the question | the question text | write or read outside the allowlist, run code, exhaust the host |
| The model | plans, SQL, operator replies, tool calls, answers | modify data, read hidden columns or other tables/files, load extensions, loop forever |
| Whoever wrote a row (e.g. a review) | text inside the database | give the model instructions that change the SQL, other rows' verdicts or the data exposed |
| Whoever controls `OLLAMA_HOST` | where prompts (and rows) are sent | receive your data without your explicit opt-in |

| Threat | Control |
| --- | --- |
| Destructive or exfiltrating SQL | Four independent layers. **Guard:** sqlglot parses the text; exactly one `SELECT`/set operation of `SELECT`s; no DDL/DML/`PRAGMA`/`ATTACH`/`DETACH`/`VACUUM INTO`/`SELECT INTO`/transactions/recursive CTEs; functions from an allowlist only (no `load_extension`, `readfile`, `writefile`, `read_csv*`, `zeroblob`, `randomblob`, `printf`, `random`, `pragma_*`); allowlisted tables and columns only, no schema qualifiers, no `*` over tables with hidden columns, no alias shadowing a hidden column; ≤ 4 table refs, ≤ 4,000 chars, ≤ 800 nodes; the SQL actually executed is regenerated from the tree without comments. **Connection:** `mode=ro` URI, `query_only=ON`, `trusted_schema=OFF`, extension loading off, `SQLITE_LIMIT_ATTACHED=0`, length/depth limits. **Authorizer:** SQLite denies everything but `SELECT`, reads of allowlisted `table.column`s and allowlisted functions. **Runtime:** timeout (progress handler), row cap, cell cap. A 51-query adversarial corpus is rejected by the guard and, with the guard bypassed, by the engine (tested). |
| SQL injection through values | Values travel as named bound parameters (`:name`), type- and length-checked; the guard rejects `?`, missing and unused parameters. Code never builds SQL with f-strings (an AST test enforces it). |
| Hidden or sensitive columns | Only the catalog is visible: the schema prompt lists allowlisted columns only, with **no data rows** unless `TAG_SCHEMA_SAMPLES` is set. The guard and the authorizer refuse hidden columns (`reviews.reviewer_email` in the bundled data), and the RAG index is built through the same allowlist. |
| Prompt injection in table rows | Rows only enter prompts through `<data>` blocks, escaped (`& < >`), so a row containing `</data>` can't close the element. System prompts are constants. Without schema samples, rows never reach query synthesis, so a row can't steer the SQL. Semantic operators judge **one row per call** and parse the reply into yes/no, an integer 0-10 or one short line: an injected row can at worst flip its own verdict. **Limits:** answer generation and the agent read rows as free text, and a model that obeys injected text can still be steered in its prose (the tests show a gullible agent answering `PWNED`). What it can't do is reach data outside the allowlist: its attempt to select `reviewer_email` is refused. Treat answers about untrusted rows as untrusted. |
| Runaway cost | `TAG_MAX_LM_ROWS` per operator (refused, not truncated), `TAG_MAX_LM_CALLS` per question, `TAG_MAX_TOOL_CALLS` for the agent (`ToolCallLimitMiddleware`) plus a recursion limit, one repair round for synthesis, questions ≤ 2,000 chars. |
| Data leaving the machine | `OLLAMA_HOST` must be loopback unless `TAG_ALLOW_REMOTE_OLLAMA=1`. Hosted providers are opt-in and need your own key. LangSmith tracing stays off unless you set `LANGSMITH_TRACING`. |
| Secrets | No key is read except `ANTHROPIC_API_KEY`, and only by the optional adapter. Keys never enter prompts, traces or logs. `.env` is git-ignored. |
| Unsafe code in the package | An AST test bans `eval`, `exec`, `compile`, `__import__`, `pickle`, `marshal`, `shelve`, `subprocess`, `yaml`, `requests`, f-string SQL and any `noqa`/`nosec`/`type: ignore`. |

## 2. Dependencies

CI runs `bandit` on the package and `pip-audit` on the locked environment. Dependabot
alerts, security updates, secret scanning and private vulnerability reporting are enabled
on the repository.

## 3. Running it safely on your own data

- Point `TAG_DB` at a **copy** or a read replica, list the tables in `TAG_TABLES` and every
  sensitive column in `TAG_HIDDEN_COLUMNS`. The allowlist is the security boundary: what
  is allowlisted can be read and sent to the model.
- Leave `TAG_SCHEMA_SAMPLES=0` unless your rows are trusted.
- Keep ollama on `127.0.0.1`. If you point `OLLAMA_HOST` elsewhere, the rows you query go
  there too.
- Treat answers as drafts and check them against the SQL and trace (`-v`).
