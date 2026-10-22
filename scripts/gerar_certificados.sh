#!/usr/bin/env bash
# Creates a private CA, the cloud broker certificate and MQTT passwords for each Hub.
# Usage: bash scripts/gerar_certificados.sh <broker-hostname> house_001 [house_002 ...]
set -euo pipefail
HOST=${1:?broker hostname, e.g. vps55112.example.com}; shift
mkdir -p certs/ca certs/server certs/hub cloud_data/mosquitto
cd certs
[ -f ca/ca.key ] || openssl req -x509 -newkey rsa:4096 -nodes -days 3650 \
    -keyout ca/ca.key -out ca/ca.crt -subj "/CN=QVDE private CA"
openssl req -newkey rsa:2048 -nodes -keyout server/server.key -out server/server.csr -subj "/CN=$HOST"
printf "subjectAltName=DNS:%s" "$HOST" > server/san.ext
openssl x509 -req -in server/server.csr -CA ca/ca.crt -CAkey ca/ca.key -CAcreateserial \
    -days 825 -out server/server.crt -extfile server/san.ext
cp ca/ca.crt server/ca.crt; cp ca/ca.crt hub/ca.crt
cd ..
touch cloud_data/mosquitto/passwd
for USERNAME in qvde-ingest qvde-admin "$@"; do
  PASS=$(openssl rand -base64 18)
  docker run --rm -v "$PWD/cloud_data/mosquitto:/m" eclipse-mosquitto:2.0.20 \
      mosquitto_passwd -b /m/passwd "$USERNAME" "$PASS"
  echo "$USERNAME  $PASS" >> certs/credenciais.txt
done
chmod 600 certs/credenciais.txt certs/ca/ca.key
echo "Done. Passwords in certs/credenciais.txt (copy each Hub's line to its .env and delete the file)."
