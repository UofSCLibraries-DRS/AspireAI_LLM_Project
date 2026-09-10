#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
frontend_dir="${script_dir}/frontend"
deploy_dir="/var/www/html"
backup_dir="/var/www/html.bak"

if [[ ! -f "${frontend_dir}/index.html" ]]; then
    echo "Error: ${frontend_dir}/index.html does not exist." >&2
    exit 1
fi

if (( EUID == 0 )); then
    privileged=()
else
    if ! command -v sudo >/dev/null 2>&1; then
        echo "Error: this deployment requires root privileges or sudo." >&2
        exit 1
    fi
    privileged=(sudo)
fi

"${privileged[@]}" rm -rf -- "${backup_dir}"

if "${privileged[@]}" test -e "${deploy_dir}"; then
    "${privileged[@]}" mv -- "${deploy_dir}" "${backup_dir}"
fi

"${privileged[@]}" mkdir -- "${deploy_dir}"

if ! "${privileged[@]}" cp -a -- "${frontend_dir}/." "${deploy_dir}/"; then
    echo "Deployment failed; restoring the previous site." >&2
    "${privileged[@]}" rm -rf -- "${deploy_dir}"
    if "${privileged[@]}" test -e "${backup_dir}"; then
        "${privileged[@]}" mv -- "${backup_dir}" "${deploy_dir}"
    fi
    exit 1
fi

echo "Frontend deployed to ${deploy_dir}."
if "${privileged[@]}" test -e "${backup_dir}"; then
    echo "Previous deployment saved at ${backup_dir}."
fi 
