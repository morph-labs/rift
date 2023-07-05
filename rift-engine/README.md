# ️🤖⚙️ Rift Code Engine

An AI-first language server for powering your personal, on-device AI software engineer. Built and maintained by [Morph](https://morph.so).

## Installation

For development:

```bash
# from this directory
pip install -e .
```

From PyPI:

```bash
pip install pyrift
```

Add in your OpenAI key:

```bash
cp .env_example .env
```

## Development

Use `conda` or `venv` to create and activate a virtual environment.

`pip install -e .`

## Running

Run the server with `python -m rift.server.core --port 7797`. This will listen for LSP connections on port 7797.

Hot reloading + debug logging setup:

```bash
# pip install watchdog if not already installed
watchmedo shell-command \
    --patterns="*.py;*.txt" \
    --recursive \
    --command='python -m rift.server.core --port 7797 --debug True'  
```

Then run `python /rift-engine/rift/llm/test.py` in separate terminal.

## Contributing
[Fork](https://docs.github.com/en/get-started/quickstart/contributing-to-projects) this repository and make a pull request.

