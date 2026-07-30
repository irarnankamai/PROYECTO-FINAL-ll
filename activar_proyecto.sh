#!/usr/bin/env bash

set -e

RAIZ="$(
    cd "$(dirname "${BASH_SOURCE[0]}")"
    pwd
)"

cd "$RAIZ"

if [ ! -f ".venv/bin/activate" ]; then
    echo "No se encontró .venv/bin/activate"
    exit 1
fi

if [ ! -f ".env.local" ]; then
    echo "No se encontró .env.local"
    exit 1
fi

source .venv/bin/activate
source .env.local

echo "Entorno del proyecto activado."
echo "API: $TAXI_API_URL"
echo "Vehículo: $TARGET_VEHICLE"

exec "${SHELL:-/bin/bash}"
