# Docker Setup for SPAC

This repository includes Docker support for both production and development workflows.

## Prerequisites

- Docker installed on your system
- Docker Compose (usually comes with Docker Desktop)
- Make (optional, but recommended)

## Quick Start

### Production Mode

Production mode creates a self-contained image with all code baked in. Use this for:
- Testing the final package
- Deployment
- Sharing with reviewers

```bash
# Build the production image
make build

# Run production container with Jupyter
make run

# Access Jupyter at http://localhost:8888
```

### Development Mode

Development mode mounts your source code as volumes, so changes are reflected immediately. Use this for:
- Active development
- Debugging
- Testing changes without rebuilding

```bash
# Start development environment
make dev

# Access Jupyter at http://localhost:8889
# Your code changes in src/ will be immediately available!
```

## Available Commands

Run `make help` to see all available commands:

```bash
make help
```

### Production Commands

- `make build` - Build production Docker image
- `make run` - Run production container with Jupyter
- `make test` - Run tests in production container
- `make rebuild` - Clean and rebuild from scratch

### Development Commands

- `make dev` - Start development environment with live code mounting
- `make dev-bash` - Start development container with bash shell (for debugging)
- `make dev-test` - Run tests in development mode

### Utility Commands

- `make stop` - Stop all running containers
- `make clean` - Remove containers and images
- `make logs` - Show container logs

## Development Workflow

1. **Start the development environment:**
   ```bash
   make dev
   ```

2. **Make changes to your code** in `src/spac/` - changes are immediately available in the container!

3. **Test your changes:**
   ```bash
   make dev-test
   ```

4. **Need to debug?** Open a shell in the container:
   ```bash
   make dev-bash
   ```

5. **When done:**
   ```bash
   make stop
   ```

## Directory Structure

- `Dockerfile` (root) - Production image (code copied at build time)
- `docker/Dockerfile.dev` - Development image (code mounted as volume)
- `docker/docker-compose.dev.yml` - Development orchestration
- `Makefile` (root) - Convenient commands for both modes

## Volumes

Both modes mount these directories:
- `./data` → `/data` (input data)
- `./results` → `/results` (output results)

Development mode additionally mounts:
- `./src` → `/home/reviewer/SCSAWorkflow/src` (source code)
- `./tests` → `/home/reviewer/SCSAWorkflow/tests` (test files)
- `./notebooks` → `/workspace` (Jupyter notebooks)

## Ports

- Production: http://localhost:8888
- Development: http://localhost:8889

## Manual Docker Commands

If you prefer not to use Make:

### Production
```bash
# Build
docker build -t spac:latest .

# Run
docker run -d --name spac-prod -p 8888:8888 \
  -v $(pwd)/data:/data \
  -v $(pwd)/results:/results \
  spac:latest
```

### Development
```bash
# Build
docker build -f docker/Dockerfile.dev -t spac:dev .

# Run
docker-compose -f docker/docker-compose.dev.yml up -d
```

## Troubleshooting

### Port already in use
If you get a port conflict, stop the other container:
```bash
make stop
```

### Changes not reflected in dev mode
Make sure you're running in development mode (`make dev`) and editing files in the `src/` directory.

### Need to rebuild
If you've changed dependencies or environment.yml:
```bash
make rebuild  # Production
# or
docker-compose -f docker/docker-compose.dev.yml build --no-cache  # Development
```

## CI/CD Integration

For CI/CD pipelines, use the production mode:
```bash
docker build -t spac:latest .
docker run --rm spac:latest /opt/conda/envs/spac/bin/pytest tests/ -v
```
