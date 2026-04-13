#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
FRONTEND_DIR="$WORKSPACE/frontend"
DEV_HOME="/home/dev"
AUTHORIZED_KEYS_SRC="$WORKSPACE/docker/ssh/authorized_keys"
AUTHORIZED_KEYS_DST="$DEV_HOME/.ssh/authorized_keys"
LOCKFILE_HASH_FILE="$FRONTEND_DIR/node_modules/.package-lock.sha256"

mkdir -p /run/sshd "$DEV_HOME/.ssh" "$FRONTEND_DIR/node_modules"
chown -R dev:dev "$DEV_HOME/.ssh" "$FRONTEND_DIR/node_modules"
chmod 700 "$DEV_HOME/.ssh"

if [ -f "$AUTHORIZED_KEYS_SRC" ]; then
  cp "$AUTHORIZED_KEYS_SRC" "$AUTHORIZED_KEYS_DST"
else
  cp /opt/dev/authorized_keys.example "$AUTHORIZED_KEYS_DST"
  echo "No docker/ssh/authorized_keys file found. SSH login will stay disabled until you add one."
fi

chown dev:dev "$AUTHORIZED_KEYS_DST"
chmod 600 "$AUTHORIZED_KEYS_DST"

ssh-keygen -A >/dev/null

if [ -f "$FRONTEND_DIR/package-lock.json" ]; then
  current_hash="$(sha256sum "$FRONTEND_DIR/package-lock.json" | awk '{print $1}')"
  installed_hash=""
  if [ -f "$LOCKFILE_HASH_FILE" ]; then
    installed_hash="$(cat "$LOCKFILE_HASH_FILE")"
  fi
  if [ ! -d "$FRONTEND_DIR/node_modules/next" ] || [ "$current_hash" != "$installed_hash" ]; then
    su - dev -c "cd '$FRONTEND_DIR' && npm ci"
    printf '%s' "$current_hash" > "$LOCKFILE_HASH_FILE"
    chown dev:dev "$LOCKFILE_HASH_FILE"
  fi
fi

exec /usr/bin/supervisord -n -c /etc/supervisor/conf.d/supervisord.conf
