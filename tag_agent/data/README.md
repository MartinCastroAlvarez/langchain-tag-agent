# Bundled data

All of it is **fictional and self-made** for this project, released under the project's
MIT license. Films, directors, reviewers and numbers are invented; the setting cities are
real places so that questions can need world knowledge (is Lisbon a European capital?).

| File | What it is |
| --- | --- |
| `movies.json` | The source of truth: 28 films and 23 reviews. Edit this, then run `python -m tag_agent.dataset`. |
| `movies.sqlite` | Built from `movies.json`. CI checks it is in sync (`python -m tag_agent.dataset --check`). |
| `bench.json` | 13 TAG-Bench-style questions (match, comparison, ranking, aggregation) with gold plans and answers. |
| `oracle.json` | The fake model's hand-written labels for the benchmark's semantic criteria. Not used by real models. |
| `adversarial.json` | 51 malicious queries the SQL guard must reject. |

Two rows are deliberately hostile:

- `reviews.reviewer_email` holds placeholder addresses (`<handle>@example.com`). The column
  is **not allowlisted**: the model never sees it and every layer refuses to read it.
- Review 6 contains a prompt injection with a `</data>` breakout attempt, an
  "ignore all previous instructions" line and a request for every reviewer email. The
  tests prove it changes neither the instructions, the SQL nor the data exposed.
