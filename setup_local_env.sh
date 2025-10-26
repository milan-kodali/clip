# set up local environment
# run: `source setup_local_env.sh`

# check if uv is installed; install it if not
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
# check if .venv dir exists; create a venv if not
[ -d ".venv" ] || uv venv
# install the repo dependencies
uv sync
# activate venv so that `python` uses the project's venv instead of system python
source .venv/bin/activate
# start jupyter (uncomment if not using vscode)
# jupyter notebook