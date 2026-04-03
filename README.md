# WhisperBridgeCast

## Packaging

This project is pip-installable via `pyproject.toml` and exposes two console commands:

- `whisper_server`
- `captioner_gui`

### Install

Install editable (development) mode:

```bash
pip install -e .
```

Install with optional backend dependencies:

```bash
pip install -e .[optional]
```

Install with all optional extras:

```bash
pip install -e .[all]
```

### Run

Start server:

```bash
whisper_server --help
whisper_server
```

Start GUI:

```bash
captioner_gui --help
captioner_gui
```

### Notes

- Warmup audio files are packaged from `app/server/data/*.wav`.
- Relative warmup paths are resolved by the server with fallbacks, so running from different working directories is supported.
