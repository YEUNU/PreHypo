#!/bin/bash

# Shared shell helpers for local service orchestration.

# Resolve the Python interpreter. Prefers the project-local .venv (created with
# `uv venv --python 3.12 .venv`), so the run scripts work for a fresh clone
# without the user having to `source .venv/bin/activate` first. Override with
# the PYTHON_BIN env var. Falls back to system python3/python.
# Usage: PYTHON_BIN="$(resolve_python "$SCRIPT_DIR")" || exit 1
resolve_python() {
    local script_dir="${1:-$(pwd)}"
    if [ -n "${PYTHON_BIN:-}" ] && [ -x "${PYTHON_BIN}" ]; then
        echo "${PYTHON_BIN}"; return 0
    fi
    if [ -x "${script_dir}/.venv/bin/python" ]; then
        echo "${script_dir}/.venv/bin/python"; return 0
    fi
    if command -v python3 >/dev/null 2>&1; then command -v python3; return 0; fi
    if command -v python  >/dev/null 2>&1; then command -v python;  return 0; fi
    echo "ERROR: no Python interpreter found. Create the env first:" >&2
    echo "  uv venv --python 3.12 .venv && VIRTUAL_ENV=.venv uv pip install -e ." >&2
    return 1
}

wait_for_server() {
    local url="$1"
    local name="$2"
    local success_codes="${3:-200|401|405}"
    local max_attempts="${4:-300}"
    local attempt=0
    local check_url="$url"
    local curl_args=(-s -o /dev/null -w "%{http_code}" --max-time 2)

    if [[ "$url" == */v1/models ]]; then
        # For local model readiness checks, prefer the lightweight health endpoint.
        check_url="${url%/v1/models}/health"
    elif [ -n "${VLLM_API_KEY:-}" ] && [ "${VLLM_API_KEY}" != "EMPTY" ]; then
        curl_args+=(-H "Authorization: Bearer ${VLLM_API_KEY}")
    fi

    echo "Wait for $name ($check_url)..."
    while [ "$attempt" -lt "$max_attempts" ]; do
        if curl "${curl_args[@]}" "$check_url" | grep -qE "$success_codes"; then
            echo " ✅ $name is Ready!"
            return 0
        fi
        printf "."
        sleep 5
        attempt=$((attempt + 1))

        if [ $((attempt % 12)) -eq 0 ]; then
            echo " (Waiting for ${name}... $((attempt * 5))s elapsed)"
        fi
    done

    echo ""
    echo " ❌ ERROR: $name failed to start after $((max_attempts * 5 / 60)) minutes."
    return 1
}
