# ScholarAgent Docker helper

Common flows for teammates after cloning the repo.

## 1) Infra only (Postgres / Redis / Milvus)

```bash
cp .env.example .env
docker compose pull
docker compose up -d
```

Services:

| Service | Host port | Notes |
|---------|-----------|-------|
| Postgres | 5432 | db=`huawei-paper-search`, user/pass=`postgres/postgres` |
| Redis | 6379 | optional cache |
| Milvus | 19530 | needs etcd + minio |
| MinIO console | 9001 | minioadmin / minioadmin |

Local backend `.env` can point to:

```env
POSTGRES_URI=postgresql://postgres:postgres@127.0.0.1:5432/huawei-paper-search
MILVUS_URI=http://127.0.0.1:19530
REDIS_HOST=127.0.0.1
REDIS_PORT=6379
```

## 2) Full stack (infra + backend + frontend)

Fill API keys in `.env`:

```env
ZHIPUAI_API_KEY=...
TAVILY_API_KEY=...
SERPER_API_KEY=...
OPENALEX_API_KEY=...
NCBI_API_KEY=...
NCBI_TOOL=huawei-paper-search
NCBI_EMAIL=...
```

Then:

```bash
docker compose --profile app up -d --build
```

- Frontend: http://localhost:8080
- Backend docs: http://localhost:8000/docs

First backend start resolves the fixed BGE revisions from `.env` and downloads them into the
`hf_cache` volume. The volume survives container rebuilds. Do not copy model weights into Git
or bake them into the application image. CPU deployment should reserve at least 12 GB RAM;
16 GB is safer when both models and all infrastructure services run on one machine.

Pinned revisions:

```text
BAAI/bge-m3                 5617a9f61b028005a4858fdac845db406aefb181
BAAI/bge-reranker-v2-m3     953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e
```

For an offline host, mount verified model snapshots instead of the Hub cache and set:

```env
BGE_M3_MODEL_PATH=/models/bge-m3
BGE_RERANKER_MODEL_PATH=/models/bge-reranker-v2-m3
BGE_LOCAL_FILES_ONLY=1
```

The corresponding Compose override must mount the host model directory read-only at `/models`.

## Database migrations

The backend can create tables lazily for a single-process demo. For controlled deployment,
apply the SQL files once, in numeric order, before accepting traffic:

```text
backend/migrations/000_base.sql
backend/migrations/001_evidrank.sql
backend/migrations/002_output_artifacts.sql
```

## Stop / cleanup

```bash
docker compose --profile app down
# remove data volumes too:
docker compose --profile app down -v
```
