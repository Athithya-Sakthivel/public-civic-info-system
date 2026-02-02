docker run -d \
  --name pgvector \
  -p 5432:5432 \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=haystack \
  -v pgvector-data:/var/lib/postgresql/data \
  ankane/pgvector:v0.5.1
  