
# Commands for Cloud Management

## 1. Maintanance from Desktop/Windows

### 1.1 Merge in the repository folder. Apply patch, branch feature/care-contexts-cloud

```
cd C:\...\QVDE
git status
git checkout main && git pull
git checkout -b feature/care-contexts-cloud
git am $HOME\Downloads\qvde_contexts_cloud.patch
*** (herein, a change of the token of InfluxDB in docker-compose.yml, with self commit)
git push -u origin feature/care-contexts-cloud
git log --oneline -2 # should display "Add care contexts, reactive Hub service .... "

Note: After, in GitHub, open pull request of the branch to the main and do the merge. Then, the main receives all.
```

## 1.2) Update docker-compose.yml 

```
change lines of the token and org by ${INFLUXDB_TOKEN} and ${INFLUXDB_ORG}

(Get-Content docker-compose.yml) `
  -replace '- INFLUXDB_TOKEN=.*', '- INFLUXDB_TOKEN=${INFLUXDB_TOKEN}' `
  -replace '- INFLUXDB_ORG=.*',   '- INFLUXDB_ORG=${INFLUXDB_ORG}' |
  Set-Content docker-compose.yml

git diff docker-compose.yml

git add docker-compose.yml
git commit -m "Read InfluxDB token from .env"
git push -u origin feature/care-contexts-cloud


Update with patch

cd C:\Users\Eduardo\QVDE
git checkout main
git pull
git am $HOME\Downloads\qvde_ingest_reconnect.patch
git push

na VM
cd ~/Projects/QVDE
git pull
docker compose --env-file .env.cloud -f docker-compose.cloud.yml up -d --build ingest
docker compose --env-file .env.cloud -f docker-compose.cloud.yml logs --tail 5 ingest

```

Nwith buttom Compare & pull request. Create pull request and do the merge. Or, open
https://github.com/edbkei/QVDE/compare/main...feature/care-contexts-cloud
Go to New pull request, click three dots at the line feature/care-contexts-cloud

at powershell:
git checkout main
git pull
git log --oneline -3
## 2. Maintenance from Cloud/LOCAWEB

### 2.1 Update .env.cloud

```cp .env.cloud.example .env.cloud
nano .env.cloud 

 Change password inside of .env.cloud
openssl rand -base64 18 (one per variable)

 Secrets
- Password of MQTT of each hub (generated in certs/credenciais.txt) and the certificates staay in the folder certs/ and cloud_data/


Check red light off & fan stopped.

```

### 2.2 Prepare to receive the repository

### a) Check the tools:

```
git --version
docker --version
docker compose version

Note: If needed, issue: sudo apt-get install docker-compose-plugin
```
### b) Clone the repository:
```
cd ~/Projects
git clone https://github.com/edbkei/QVDE.git
cd QVDE
git checkout feature/care-contexts-cloud

Note: On next updates, it should be enough cd ~/Projects/QVDE && git pull
```
### c) Open a broker port:
```
The Hub will conect to the port 8883, regarding to MQTT with TLS. Allow firewall as:
sudo ufw status 
sudo ufw allow 8883/tcp

Note: Pstgres, and Grafana port should not be opened. Grafana port is accessible only by SSH tunnel.
```
### d) Certificate
 
Check open ports to internet, in case there any origin (Anywhere). 

Port   Service                    Recommendation
5432   PostgreSQL VM native       Close, if no external usage
1883   MQTT without cryptography  Close (8883 replaces it)
8086   InfluxDB                   Close
3000   Grafana                    Close (only accessible by SSH tunnel)

sudo ufw delete allow 5432/tcp
sudo ufw delete allow 1883/tcp
sudo ufw delete allow 8086/tcp
sudo ufw delete allow 3000/tcp


As the TLS certificate is issued for a name, and the Hub needs to connect using exactly this name:
```
hostname -f

Config:

cp .env.cloud.example .env.cloud && nano .env.cloud              # password here
bash scripts/gerar_certificados.sh <nome-da-vm> house_001
docker compose --env-file .env.cloud -f docker-compose.cloud.yml up -d
docker compose --env-file .env.cloud -f docker-compose.cloud.yml ps

Note: The step-by-step (residence registration, context publishing) is at the folder docs/CARE_CONTEXT_AND_CLOUD.md``

```

### e) Clone and configure
```
cd ~/Projects
git clone https://github.com/edbkei/QVDE.git
cd QVDE

cp .env.cloud.example .env.cloud
for v in POSTGRES_PASSWORD GRAFANA_ADMIN_PASSWORD MQTT_INGEST_PASSWORD MQTT_ADMIN_PASSWORD; do
  sed -i "s|^$v=change-me|$v=$(openssl rand -hex 16)|" .env.cloud
done
cat .env.cloud
```
### f) MQTT certificate and password
```
bash scripts/gerar_certificados.sh vps55112.publiccloud.com.br house_001

Note: the script create a CA, broker certificate and MQTT password, and record psw in certs/credenciais.txt. The line house_001
goes to .env of the Jetson

Copy qvde-ingest and qvde-admin psw of the credential files to .env.cloud

ING=$(awk '$1=="qvde-ingest"{print $2}' certs/credenciais.txt)
ADM=$(awk '$1=="qvde-admin"{print $2}' certs/credenciais.txt)
sed -i "s|^MQTT_INGEST_PASSWORD=.*|MQTT_INGEST_PASSWORD=$ING|; s|^MQTT_ADMIN_PASSWORD=.*|MQTT_ADMIN_PASSWORD=$ADM|" .env.cloud
grep -c "MQTT_.*_PASSWORD=." .env.cloud     # deve dar 2
```

### g) Up services

```
docker compose --env-file .env.cloud -f docker-compose.cloud.yml up -d
docker compose --env-file .env.cloud -f docker-compose.cloud.yml ps


# Grafana (usuário 472)
sudo chown -R 472:0 cloud_data/grafana

# Mosquitto (usuário 1883): chave, senhas, ACL e dados
sudo chown 1883:1883 certs/server/server.key cloud_data/mosquitto/passwd cloud/mosquitto/acl
sudo chmod 600 certs/server/server.key cloud_data/mosquitto/passwd cloud/mosquitto/acl
sudo chown -R 1883:1883 cloud_data/mosquitto/data

docker compose --env-file .env.cloud -f docker-compose.cloud.yml restart grafana mosquitto
docker compose --env-file .env.cloud -f docker-compose.cloud.yml ps

docker compose --env-file .env.cloud -f docker-compose.cloud.yml logs --tail 5 mosquitto ingest
Note: should appear: mosquitto version 2.0.20 running. And in the ingest: subscribed to qvde/v1/+/up/+.
```
### h) Register a residence and publish the contexts

```
Create a shortcut:
-----------------
cd ~/Projects/QVDE
alias twin='docker compose --env-file .env.cloud -f docker-compose.cloud.yml run --rm twin-admin'

Then:
----
twin registrar-residencia --house house_001 --paciente PAC-0001 \
     --contextos risco_queda,pressao_arterial,hipotensao_ortostatica_queda
 
Result: House_001 -> PAC-0001    
```
```   
     
     
twin publicar-contextos --dir contexts
Result: 3 contexts, each one with v1

twin definir-desejado --house house_001

result: desired state v1 for house_001

```   
```   

twin status --house house_001


Result:

"desejado", fulfilled

"reported": null, when Hub not conected yet

"sincronizado": false

"sincronizado": true, when Jetson is connected

``` 
 
```   
Note: Alias only is valid for current SSH session, if wanted to keep up, do:
echo "alias twin='...'" >> ~/.bashrc
```   