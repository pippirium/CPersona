# Fork-local development policy

- This checkout is the user's experimental fork: `pippirium/CPersona`.
- Do not create pull requests against `Cloto-dev/CPersona` or other upstream repositories.
- Do not push to upstream. Keep upstream fetch-only; its local push URL is intentionally disabled.
- Do not publish or push local work unless the user requests it.
- Embedding PoC instructions are in `poc/embedding/README.md`.
- Keep experiments isolated from the user's production memory database and MCP configuration.
- Normal recall/search must return text and asset references only, never automatically load or return images.
