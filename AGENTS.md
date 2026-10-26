# Brain-AI (brain-automl): agent guide

Multimodal decision-intelligence / AutoML engine: decides and executes multimodal ML pipelines (`PipelineSpec = modalities, granularity, fusion, model backend, hyperparameters`). Open source; published on PyPI as `brain-automl`; basis of the owner's PhD.

Requirements: [PRODUCT.md](PRODUCT.md). Verification gate: `bash scripts/test.sh`.

## Layout and commands

- `src/brain_ai/`: the package; `tests/` the test suite; `examples/` notebooks.
- venv: `.venv` (Python 3.12) with `pip install -r requirements.txt -e .`.

## Project-specific rules

- Public, open-source repository: no private data, keys or unpublished research results.
- Keep the public API (`PipelineSpec`, adapters) backward compatible; PyPI releases are owner-approved.

## Rules for every agent (Claude Code, Codex, Supervisor workers)

- **Requirements:** `PRODUCT.md` says what the product must do; build to its acceptance criteria. If a request conflicts with it, say so instead of guessing.
- **Done means the gate passes:** `bash scripts/test.sh` is the verification gate the Supervisor runs in each task worktree. Keep it green; add tests beside the code you change.
- **No fake data:** never ship placeholder or demo values in the UI; use empty states.
- **Never invent facts** in docs, marketing or store listings; claim only what exists.
- **Git:** commit only when asked; never force-push, rewrite history or delete branches (archive instead). The Supervisor merges verified work to the default branch.
- **Secrets:** `.env` files and keys stay out of git.
- **Test-user data:** the developers' own data (test users) lives in the private ExoMind repo (`exomind/testdata/users/<person>/`). Never commit it to this repo, bundle it into an app, or ship it in a build, store upload or deployment (policy: `exomind/testdata/README.md`).
- **Storage:** virtualenvs, caches, models and build output stay inside this repo on the external SSD (`/Volumes/MacSSD`); the internal disk is nearly full.
- **Cost:** Claude/OpenAI models only through the owner's Claude Code / Codex subscriptions, never per-token via OpenRouter.
- Workspace rules: `/Volumes/MacSSD/01_Projects/SolarpunkWorks/CLAUDE.md` and `workspace.yaml`.
