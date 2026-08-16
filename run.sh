#!/bin/bash
# Convenience wrapper to run algotrader with virtual environment

# Resolve symlinks so SCRIPT_DIR is the real repo, not the install location.
# The skill is normally installed as symlinks under ~/.claude/skills/algotrader;
# without this, invoking the symlinked run.sh builds a second 275MB venv there.
SOURCE="${BASH_SOURCE[0]}"
while [ -L "$SOURCE" ]; do
    LINK_DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"
    SOURCE="$( readlink "$SOURCE" )"
    [[ $SOURCE != /* ]] && SOURCE="$LINK_DIR/$SOURCE"
done
SCRIPT_DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"
VENV_PYTHON="$SCRIPT_DIR/venv/bin/python"

# Check if venv exists
if [ ! -f "$VENV_PYTHON" ]; then
    echo "⚠️  Virtual environment not found. Creating..."
    python3 -m venv "$SCRIPT_DIR/venv"
    "$SCRIPT_DIR/venv/bin/pip" install -r "$SCRIPT_DIR/requirements.txt"
fi

# Run algotrader with all arguments
"$VENV_PYTHON" "$SCRIPT_DIR/algotrader.py" "$@"
