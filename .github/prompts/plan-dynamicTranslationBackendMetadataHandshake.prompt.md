## Plan: Dynamic Translation Backend Metadata Handshake

Expand `translator_initialized` status payload to include server-provided backend capabilities, then drive client translation UI from that metadata (engine/provider first, target language second and filtered by selected speech language + provider). Use a unified backend schema with a default `provider: "local"` for local backends, including Whisper as a pseudo-local backend.

**Steps**
1. Phase 1: Define protocol contract for `translator_initialized` payload (*blocks all later steps*).
2. Add a normalized `value.backends` schema that includes, per backend: backend `name`, `providers` (or model list represented as providers), and supported source/target language information.
3. Include `status` unchanged in `value.status` for backward compatibility with existing status handling.
4. Define canonical language representation and directionality rules (source->target pairs), plus how “all languages”/unknown capabilities are represented for broader capability backends.
5. Phase 2: Add server-side capability functions per backend (*depends on Phase 1*).
6. Introduce per-backend capability function(s) in translator backend classes to return supported pairs and provider/model metadata.
7. Ensure each backend can expose source and target language support; for local backends expose a single provider named `local`.
8. Add Whisper as a pseudo-local backend in metadata generation so UI has one unified backend list.
9. For online translators, expose provider list and provider-level language capabilities from backend capability functions (not hardcoded UI lists).
10. For broader capability backends where exact pairs are not enumerable, define a clear capability mode in metadata (for example explicit pairs vs broad/unknown) and keep schema consistent.
11. Phase 3: Emit metadata from server status flow (*depends on Phase 2*).
12. Expand `translator_initialized` message `value` dict to include `backends` using the normalized schema while preserving existing `status` field.
13. Build metadata once during translator init path so it reflects runtime-available backends/providers and can be reused safely.
14. Keep message resilient: if one backend capability lookup fails, include remaining backends and attach minimal error detail for observability without breaking initialization.
15. Phase 4: Client ingest and state model (*depends on Phase 3*).
16. Extend client status message parsing to read `value.backends` when `status == translator_initialized` and cache it in UI state.
17. Add fallback behavior: if `backends` missing (older server), continue with current static lists to preserve compatibility.
18. Add internal helper methods for querying cached capabilities: available engines, providers for engine, and targets for (engine, provider, source speech language).
19. Phase 5: Rework UI population order and filtering (*depends on Phase 4*).
20. Populate Translation Engine combo from server metadata first; for online translators populate Provider combo from selected engine metadata.
21. Populate Target Language combo only after engine/provider resolution, filtered by currently selected Whisper Speech language.
22. Bind Speech language change event so target-language options are recomputed immediately when source language changes.
23. On engine/provider/source changes, preserve current selection if still valid; otherwise select first valid target and update dependent controls.
24. Keep provider control hidden/disabled for local backends (single `local` provider) while still using unified internal schema.
25. Phase 6: Validation and connect-time guardrails (*depends on Phase 5*).
26. Before connect/start, validate selected (engine, provider, source, target) against metadata; block invalid combinations with a clear UI error.
27. Ensure params sent to server use selected provider/engine values that correspond to metadata entries.
28. Phase 7: Regression safety and cleanup (*parallel with Phase 6 after core wiring*).
29. Keep legacy hardcoded sets only as fallback path; route normal connected flow through metadata-driven logic.
30. Add logging around metadata receipt/parsing and invalid-pair prevention to simplify troubleshooting.

**Relevant files**
- `c:/Users/milan/source/repos/WhisperBridgeCast/whisper_server.py` — extend translator init status payload, add backend capability function contract, implement metadata assembly, include Whisper pseudo-backend.
- `c:/Users/milan/source/repos/WhisperBridgeCast/captioner_gui.py` — ingest `translator_initialized.value.backends`, refactor translation tab population order, add filtering on speech language/provider, add fallback path.
- `c:/Users/milan/source/repos/WhisperBridgeCast/net_common.py` — verify no protocol changes needed beyond payload expansion (JSON transport should already support it).
- `c:/Users/milan/source/repos/WhisperBridgeCast/todo.txt` — optional: mark backend-metadata handshake item done after implementation.

**Verification**
1. Protocol verification: connect client and inspect incoming `status=translator_initialized` payload includes `value.backends` with expected fields for each backend.
2. UI order verification: engine/provider controls populate before target language; target options update when speech language changes.
3. Filtering verification: for each backend/provider, target list matches metadata for selected source language.
4. Whisper verification: Whisper appears as backend with provider `local`, provider UI hidden for it, and target filtering still works.
5. Compatibility verification: run client against old server (no backends payload) and confirm existing static UI behavior remains functional.
6. Invalid pair verification: select unsupported combination and confirm client blocks connect/start with actionable error.
7. Runtime verification: successful connection sends chosen engine/provider/language params and server reaches ready state.

**Decisions**
- Embed backend metadata in `translator_initialized.value.backends` (no separate metadata message).
- Include Whisper as pseudo backend in server metadata.
- Use `provider: local` for local backends to unify schema.
- Target broad backend capability reporting (not limited to current app language list), with explicit representation when exact pairs are not enumerable (query supported languages when possible for online providers using deep-translator library, if not possible then use current app language list; for local backends, use the curren app language list, which shall be manually expanded later when needed).

**Scope boundaries**
- Included: server capability discovery contract, status payload expansion, client dynamic population/filtering, backward-compatible fallback.
- Excluded: redesign of translation inference algorithms, adding new translation models/providers themselves, unrelated UI redesign.

**Further Considerations**
1. Capability granularity: for broad/unknown-support backends, decide whether to present unrestricted target list or only app-known languages until explicit pairs are available.
2. Metadata freshness: decide if capability metadata is static per connection (current plan) or refreshable at runtime.
3. Error UX: decide whether invalid pair errors should be modal dialog or inline validation near translation controls.
