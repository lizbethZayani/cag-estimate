# Docker Setup Guide for CAG Estimate

This document explains how to run the CAG Estimate API using Docker and Docker Compose.

## Prerequisites

- [Docker](https://www.docker.com/get-started) (version 20.10+)
- [Docker Compose](https://docs.docker.com/compose/install/) (version 1.29+)
- Anthropic API Key (required)
- OpenAI API Key (optional; needed for the semantic cache, moderation and the fallback model)

## Quick Start

### 1. Configure Environment

Copy the Docker environment template:
```bash
cp .env.docker .env
```

Edit `.env` and add your Anthropic API key:
```bash
ANTHROPIC_API_KEY=your_api_key_here
```

### 2. Build and Run with Docker Compose

**Development mode (with auto-reload):**
```bash
docker-compose up --build
```

The API will be available at: `http://localhost:8000`

**Production mode (stable, no hot reload):**
```bash
COMPOSE_FILE=docker-compose.yml docker-compose up --build
```

### 3. Verify the Service

Check health status:
```bash
curl http://localhost:8000/health
```

Access API documentation:
```
http://localhost:8000/docs      # Swagger UI
http://localhost:8000/redoc     # ReDoc
```

## Docker Commands

### Build the Image

```bash
docker build -t cag-estimate:latest .
```

### Run Container Directly (without Compose)

```bash
docker run -p 8000:8000 \
  -e ANTHROPIC_API_KEY=your_key \
  -e LLM_PROVIDER=anthropic \
  -e LLM_MODEL=claude-haiku-4-5-20251001 \
  cag-estimate:latest
```

### View Logs

```bash
# Follow logs in real-time
docker-compose logs -f api

# View last 100 lines
docker-compose logs --tail=100 api
```

### Stop Services

```bash
docker-compose down
```

### Remove Everything (containers, volumes, networks)

```bash
docker-compose down -v
```

### Rebuild Image

```bash
docker-compose build --no-cache api
```

## Docker Compose File Structure

### Main Configuration (`docker-compose.yml`)

- **Service:** `redis`
  - Image: `redis/redis-stack:7.4.0-v0` (Redis with RediSearch; required by the semantic cache, plain `redis:7-alpine` is not enough)
  - Port: `6379:6379`
  - Data volume: `cag-estimate-redis-data`
  - Used for the exact cache and the semantic (vector) cache
- **Service:** `api`
  - Port: `8000:8000`
  - Waits for `redis` to be healthy; `REDIS_URL` is fixed to `redis://redis:6379/0`
  - Health checks enabled
  - Auto-restart on failure
  - Network isolation
  - Non-root user (appuser, uid 1000)

- **Service:** `postgres` (container `estimator-web-postgres`)
  - Image: `postgres:16-alpine`; user, password `postgres`, database `estimator_web_development`
  - **Not published to the host**; reachable only on the compose network (`postgres:5432`)
  - Data volume: `estimator-web-pg-data`
  - Stores the Rails UI estimation history
- **Service:** `estimator-web` (container `estimator-web`)
  - Built from `../estimator-web`; port `3000:3000`
  - `ESTIMATOR_API_BASE_URL=http://api:8000`, `ESTIMATOR_AI_TIMEOUT=180`, `RAILS_ENV=development`
  - Waits for `postgres` to be healthy; runs `bin/rails db:prepare && bin/dev`
  - Source bind-mounted from `../estimator-web`; named volume `estimator-web-bundle` holds the installed gems (`/usr/local/bundle`)
  - Does not depend on `api` at startup; estimates fail with a 503 message until the API is up

This image (`redis-stack`) bundles RedisInsight on port 8001, but the compose file only publishes `6379`. To use the RedisInsight UI add `- "8001:8001"` to the `redis` ports.

### Rails web UI

Start (or rebuild) the UI with its dependencies:
```bash
docker compose up -d --build estimator-web postgres api redis
```

Open http://localhost:3000 (UI) and http://localhost:8000 (API). Stop only the UI and its database, or everything:
```bash
docker compose stop estimator-web postgres
docker compose down          # all services; add -v to also delete the volumes (this erases the estimation history)
```

Run its tests and lint with `docker compose exec estimator-web bin/rails test` (also `bin/rubocop`, `bin/brakeman`). Rails development mode does not reload initializers, so run `docker compose restart estimator-web` after changing `config/initializers/*` or env vars. See [../estimator-web/README.md](../estimator-web/README.md).

If host port `6379` is already in use (for example by another Redis), the `redis` service fails to start because this compose file publishes `6379:6379`. Create a compose override file that remaps it (for example a `docker-compose.ports.yml` containing `services: { redis: { ports: !override ["6380:6379"] } }`; the `!override` tag needs a recent Docker Compose v2, otherwise a plain list would add a second mapping instead of replacing `6379:6379`) and pass it with `docker compose -f docker-compose.yml -f docker-compose.ports.yml ...`. The `api` container still reaches Redis at `redis://redis:6379/0` over the compose network.

### Development Overrides (`docker-compose.override.yml`)

Automatically applied when using `docker-compose up`. Provides:
- Debug logging
- Hot reload enabled
- Source code mounted for development
- Interactive terminal (`tty: true`)

## Environment Variables

### Required

- `ANTHROPIC_API_KEY` - Your Anthropic API key (required)

### Optional

- `LLM_PROVIDER` - LLM provider (default: `anthropic`)
- `LLM_MODEL` - Claude model to use (default: `claude-haiku-4-5-20251001`)
- `APP_ENV` - Application environment: `development` or `production` (default: `production`)
- `LOG_LEVEL` - Logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR` (default: `INFO`; `docker-compose.override.yml` sets `DEBUG`)
- `OPENAI_API_KEY` - Enables moderation, the fallback model and the semantic-cache embeddings
- `FALLBACK_MODEL` - Fallback model (default: `gpt-4o-mini`)
- `CACHE_TTL_SECONDS` - Exact-cache TTL (default: `86400`)

Semantic cache (all passed through to the `api` container by `docker-compose.yml`):

- `EMBEDDING_MODEL` - Embedding model (default: `text-embedding-3-small`)
- `SEMANTIC_CACHE_ENABLED` - `true` or `false` (default: `true`)
- `SEMANTIC_CACHE_THRESHOLD` - Minimum cosine similarity for a hit (default: `0.90`)
- `SEMANTIC_CACHE_TTL` - Entry TTL in seconds (default: `86400`)
- `SEMANTIC_CACHE_LOG_ONLY` - Log would-be hits without serving them (default: `false`)

Without `OPENAI_API_KEY` the semantic cache is disabled and the API keeps working. `.env.example` does not list the semantic-cache variables; set them in your `.env` or the shell if you need non-default values.

## Docker Image Details

### Base Image
- `python:3.11-slim` - Minimal Python 3.11 image

### Multi-stage Build
1. **Builder stage:** Compiles dependencies and wheels
2. **Runtime stage:** Only includes runtime dependencies

### Optimizations
- Non-root user for security
- Health checks configured
- Minimal image size (~500MB)
- Layer caching for faster builds

### Exposed Ports
- `8000` - API service

## Running Tests in Docker

### Run the Test Suite

The compose file mounts `./tests` read-only, but dev dependencies are not installed in the image; run the suite on the host instead:

```bash
uv run pytest
```

For an end-to-end check against the running containers see [VERIFICATION.md](./VERIFICATION.md).

### Access Container Shell

```bash
docker-compose exec api /bin/bash
```

## Troubleshooting

### Port Already in Use

Change the port mapping in `docker-compose.yml`:
```yaml
ports:
  - "8001:8000"  # Map to port 8001 instead
```

Then access at: `http://localhost:8001`

### API Key Not Found

Ensure your `.env` file has the correct API key:
```bash
cat .env | grep ANTHROPIC_API_KEY
```

### Container Won't Start

Check logs:
```bash
docker-compose logs -f api
```

### Permission Denied Errors

The container runs as non-root user `appuser` (uid 1000). Ensure volume permissions are correct:
```bash
chmod 755 src/ tests/
```

## Production Deployment

### Using Production Configuration

```bash
# Use only the main compose file (no overrides)
COMPOSE_FILE=docker-compose.yml docker-compose up -d
```

### Docker Swarm Deployment

```bash
# Initialize swarm mode
docker swarm init

# Deploy stack
docker stack deploy -c docker-compose.yml cag-estimate
```

### Kubernetes Deployment

Convert docker-compose to Kubernetes manifests:
```bash
kompose convert -f docker-compose.yml
```

## Performance Tips

1. **Use .dockerignore** - Reduces build context size
2. **Multi-stage builds** - Reduces final image size
3. **Layer caching** - Keep dependencies stable, code at the top
4. **Health checks** - Enable automatic restart on failure

## Security Considerations

✅ **Implemented:**
- Non-root user (appuser)
- Minimal base image
- No unnecessary dependencies
- Health checks
- Environment variable isolation

✅ **Recommendations:**
- Use secrets management (Docker secrets, Kubernetes secrets)
- Scan image for vulnerabilities: `docker scan cag-estimate`
- Use private registry for deployment
- Rotate API keys regularly

## Docker Compose Services

### Current Services

- **redis** - Redis Stack (exact cache + semantic cache)
- **api** - CAG Estimate FastAPI application
- **postgres** - PostgreSQL 16 for the Rails UI history (not published to the host)
- **estimator-web** - Rails UI on port 3000

## Example: Full Workflow

```bash
# 1. Clone repository
git clone https://github.com/lizbethZayani/cag-estimate.git
cd cag-estimate/cag-estimate

# 2. Configure environment
cp .env.docker .env
# Edit .env and add your API key

# 3. Build and start
docker-compose up --build

# 4. In another terminal, test
curl http://localhost:8000/health

# 5. View logs
docker-compose logs -f api

# 6. Run estimation
curl -X POST http://localhost:8000/api/v1/estimate \
  -H "Content-Type: application/json" \
  -d '{
    "transcription": "Your meeting transcription...",
    "hourly_rate": 40
  }'

# 7. Stop services
docker-compose down
```

## References

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Python Docker Best Practices](https://docs.docker.com/language/python/build-images/)
- [FastAPI Docker Deployment](https://fastapi.tiangolo.com/deployment/docker/)

---

**For more information, see the main [README.md](./README.md) and [VERIFICATION.md](./VERIFICATION.md)**
