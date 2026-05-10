---
description: Implement a feature, open a PR, and auto-merge when CI is green
argument-hint: <feature description>
---

# /ship — autonomous PR workflow

Feature description: $ARGUMENTS

Follow this workflow exactly. Do not skip steps and do not push to `master`.

## 1. Pre-flight

- Run `git status` and `git branch --show-current`. If the working tree is dirty, ask the user whether to stash, commit, or abort.
- If the current branch is `master`, create a new branch named `feat/<short-kebab-slug-from-description>` and switch to it. Otherwise stay on the current feature branch.

## 2. Implement

- Implement the feature described in `$ARGUMENTS`.
- Use TodoWrite to track sub-steps if there are 3 or more.
- Edit existing files when possible. Do not create new files unless the feature requires it.
- If renaming any symbol, search project-wide first and update every call site in the same change.

## 3. Verify

- Run `pytest`. If tests fail, fix them before continuing. Do not commit failing tests.
- If the change is UI/frontend, describe what manual verification you would do (you cannot run a browser here).

## 4. Commit and push

- Stage only the files you changed (no `git add -A`).
- Commit with a concise message in the form `<verb>: <what changed>` (e.g., `fix: typo in README`).
- Push the branch with `git push -u origin <branch>`.

## 5. Open PR

- Use `mcp__github__create_pull_request` against `master` with:
  - **title**: same as the commit subject (under 70 chars)
  - **body**: a markdown body containing
    - `## Summary` — 1-3 bullets on what changed and why
    - `## Test plan` — checklist of what was tested (paste pytest output if relevant)
    - footer line: `https://claude.ai/code`
- Capture the PR number from the response.

## 6. Auto-merge

- Call `mcp__github__enable_pr_auto_merge` on the PR with `merge_method: "squash"` and `delete_branch: true`.
- If the call fails because there are no required status checks, fall back to: ask the user to confirm, then call `mcp__github__merge_pull_request` directly with `merge_method: "squash"` and `delete_branch: true`.

## 7. Report back

End your turn with one short message containing:
- The PR URL
- Whether auto-merge is armed or whether the user needs to merge manually
- Any follow-ups (e.g., "remember to pull master locally after merge")

## Guardrails

- Never `git push origin master` — it will 403 and waste time.
- Never `git rebase -i`, `git reset --hard`, or `git push --force` without explicit user approval.
- Never skip hooks (`--no-verify`).
- If `$ARGUMENTS` is empty, ask the user for a feature description before doing anything.
