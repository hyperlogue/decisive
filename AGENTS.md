# Decisive

## Committing

- Work on `main` directly for routine work; no feature branches or PRs.
- Other sessions share this tree. Before **each** staging operation, run
  `git diff --cached --name-only`. If another session has staged files, wait
  5s, then 10s, then 30s, rechecking after each wait. If the index is still
  occupied after the last wait, stop and report it; leave those files alone.
- Stage **and** commit only your paths: `git add <path>…`, verify the staged
  names are only yours, then `git commit -- <path>…`. Never use `git add -A`,
  `git add .`, `git add -u`, `git commit -a` / `-am`, or an index-wide commit.
  If staging or committing is blocked, preserve the work and report the blocker.
- Commit when the work is complete and no user decision remains. An open
  question or unexpected tradeoff leaves staging and committing to the user.
  Push only when the user requests it.
- Prefer small, self-contained commits that each build and pass their relevant
  tests; one logical change spanning crates stays together. Recheck the index
  before each commit's staging step.

Commit subjects are capitalized imperatives, at most 72 characters, with no
trailing period or forced scope (`doc:` is fine for documentation). Add a body,
wrapped around 72 columns, when the motivation or constraint is not clear from
the subject and diff; include a short verification note when checks ran.
Use no `Co-Authored-By` trailer, including one added by a harness.
