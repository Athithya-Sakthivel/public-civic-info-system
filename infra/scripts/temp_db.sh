docker pull athithya5354/civic-indexing:latest
docker network create civic-net || true
docker rm -f pgvector civic-indexing || true
docker run -d \
  --name pgvector \
  --network civic-net \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=haystack \
  -v pgvector-data:/var/lib/postgresql/data \
  ankane/pgvector:v0.5.1

docker run -d \
  --name civic-indexing \
  --network civic-net \
  --restart unless-stopped \
  -e AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID" \
  -e AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY" \
  -e AWS_REGION="ap-south-1" \
  -e S3_BUCKET="civic-data-raw-prod" \
  -e SEED_URLS="https://www.india.gov.in/my-government/schemes,https://www.myscheme.gov.in,https://www.tn.gov.in/schemes.php,https://nrega.nic.in,https://pmkisan.gov.in,https://csc.gov.in,https://www.nhm.tn.gov.in/en" \
  -e ALLOWED_DOMAINS="india.gov.in,myscheme.gov.in,tn.gov.in,nrega.nic.in,pmkisan.gov.in,csc.gov.in,nhm.tn.gov.in" \
  -e POSTGRES_HOST="pgvector" \
  -e POSTGRES_PORT="5432" \
  -e POSTGRES_DB="haystack" \
  -e POSTGRES_USER="postgres" \
  -e POSTGRES_PASSWORD="postgres" \
  athithya5354/civic-indexing:latest

sleep 200
docker logs --tail 200 civic-indexing