#!/usr/bin/env bash
# shellcheck shell=bash

ru_die() {
  echo "[Error] $*" >&2
  exit 1
}

ru_require_file() {
  local path="$1"
  local label="${2:-file}"
  [[ -f "${path}" ]] || ru_die "${label} not found: ${path}"
}

ru_require_python_modules() {
  local python_bin="$1"
  shift
  local modules=("$@")
  local imports
  imports="$(IFS=,; echo "${modules[*]}")"

  if ! "${python_bin}" -c "import ${imports}" >/dev/null 2>&1; then
    local modules_text
    modules_text="$(IFS=', '; echo "${modules[*]}")"
    echo "[Error] Missing python deps in current env (need at least ${modules_text})." >&2
    echo "        Try: conda activate <your_env>  (or install deps in this python)." >&2
    exit 1
  fi
}

ru_is_enabled() {
  local value="${1:-0}"
  case "${value}" in
    1|true|TRUE|True|yes|YES|Yes|on|ON|On)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

ru_array_append() {
  local array_name="$1"
  shift
  local quoted=""
  local value
  for value in "$@"; do
    printf -v quoted '%s %q' "${quoted}" "${value}"
  done
  eval "${array_name}+=( ${quoted} )"
}

ru_append_if_nonempty() {
  local array_name="$1"
  local flag="$2"
  local value="${3:-}"
  if [[ -n "${value}" ]]; then
    ru_array_append "${array_name}" "${flag}" "${value}"
  fi
  return 0
}

ru_append_if_enabled() {
  local array_name="$1"
  local enabled="${2:-0}"
  shift 2
  if ru_is_enabled "${enabled}"; then
    ru_array_append "${array_name}" "$@"
  fi
  return 0
}

ru_append_bool_flag() {
  local array_name="$1"
  local enabled="${2:-0}"
  local enable_flag="$3"
  local disable_flag="$4"
  if ru_is_enabled "${enabled}"; then
    ru_array_append "${array_name}" "${enable_flag}"
  else
    ru_array_append "${array_name}" "${disable_flag}"
  fi
}

ru_run() {
  printf '[Run]'
  printf ' %q' "$@"
  printf '\n'
  "$@"
}
