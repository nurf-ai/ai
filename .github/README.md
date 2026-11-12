# GitHub Actions dependencies

Actions are pinned in `workflows/actions.lock`, including transitive dependencies
and upstream repository identities. Keep this file with the workflow changes.

Install the pinned generator and refresh after changing any `uses:` reference:

```sh
gh extension install github/gh-actions-lock --pin v0.4.1
gh actions-lock
gh actions-lock --verify
```

Review changes to locked commits and repository identities before committing.
Use `gh actions-lock --relock` only when intentionally updating floating refs.
Dependabot checks Actions weekly with a seven-day cooldown and updates the lockfile.

CI checks lockfile provenance and runs zizmor 1.32.0, which supports lockfiles.
For local checks:

```sh
gh actions-lock --verify-local
uvx zizmor==1.32.0 --min-severity low --offline .github/workflows
```

The offline checks cover local configuration; `--verify` also checks upstream
provenance. The lockfile tool and zizmor versions in CI need periodic review.
