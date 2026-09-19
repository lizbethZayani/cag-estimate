# Docker Setup Guide for CAG Estimate

This document explains how to run the CAG Estimate API using Docker and Docker Compose.

## Prerequisites

- [Docker](https://www.docker.com/get-started) (version 20.10+)
- [Docker Compose](https://docs.docker.com/compose/install/) (version 1.29+)
- Anthropic API Key (required)

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

- **Service:** `api`
  - Port: `8000:8000`
  - Health checks enabled
  - Auto-restart on failure
  - Network isolation
  - Non-root user (appuser, uid 1000)

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
- `LOG_LEVEL` - Logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR` (default: `INFO`)

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

### Run Verification Tests

```bash
docker-compose exec api python tests/test_verification.py
```

### Run API Integration Tests

```bash
docker-compose exec api python tests/verify_api.py
```

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

- **api** - CAG Estimate FastAPI application

### Future Expansion

To add more services (e.g., database, cache), update `docker-compose.yml`:

```yaml
services:
  api:
    # ... existing config
  
  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_PASSWORD: password
    volumes:
      - postgres_data:/var/lib/postgresql/data
  
  redis:
    image: redis:7-alpine

volumes:
  postgres_data:
```

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
