.PHONY: help build run dev stop clean rebuild test

# Default target
help:
	@echo "SPAC Docker Management"
	@echo "======================"
	@echo ""
	@echo "Production targets:"
	@echo "  make build          - Build production Docker image"
	@echo "  make run            - Run production container with Jupyter"
	@echo "  make test           - Run tests in production container"
	@echo ""
	@echo "Development targets:"
	@echo "  make dev            - Start development environment with live code mounting"
	@echo "  make dev-bash       - Start development container with bash shell"
	@echo "  make dev-test       - Run tests in development mode"
	@echo ""
	@echo "Utility targets:"
	@echo "  make stop           - Stop all running containers"
	@echo "  make clean          - Remove containers and images"
	@echo "  make rebuild        - Clean and rebuild production image"
	@echo "  make logs           - Show container logs"

# Production build
build:
	@echo "Building production Docker image..."
	docker build -t spac:latest .

# Run production container
run:
	@echo "Starting production container with Jupyter..."
	docker run -d \
		--name spac-prod \
		-p 8888:8888 \
		-v $(PWD)/data:/data \
		-v $(PWD)/results:/results \
		spac:latest
	@echo "Jupyter available at http://localhost:8888"

# Development environment with live code mounting
dev:
	@echo "Starting development environment with live code mounting..."
	docker-compose -f docker/docker-compose.dev.yml up -d
	@echo "Development Jupyter available at http://localhost:8889"
	@echo "Source code is mounted - changes will be reflected immediately!"

# Development with bash shell
dev-bash:
	@echo "Starting development container with bash shell..."
	docker-compose -f docker/docker-compose.dev.yml run --rm --service-ports spac-dev bash

# Run tests in development mode
dev-test:
	@echo "Running tests in development mode..."
	docker-compose -f docker/docker-compose.dev.yml run --rm spac-dev \
		/opt/conda/envs/spac/bin/pytest tests/ -v

# Run tests in production container
test:
	@echo "Running tests in production container..."
	docker run --rm spac:latest \
		/opt/conda/envs/spac/bin/pytest tests/ -v

# Stop all containers
stop:
	@echo "Stopping containers..."
	-docker stop spac-prod 2>/dev/null || true
	-docker-compose -f docker/docker-compose.dev.yml down 2>/dev/null || true

# Clean up containers and images
clean: stop
	@echo "Cleaning up containers and images..."
	-docker rm spac-prod 2>/dev/null || true
	-docker rmi spac:latest 2>/dev/null || true
	-docker-compose -f docker/docker-compose.dev.yml down -v 2>/dev/null || true

# Rebuild from scratch
rebuild: clean build
	@echo "Rebuild complete!"

# Show logs
logs:
	@echo "=== Production logs ==="
	-docker logs spac-prod 2>/dev/null || echo "No production container running"
	@echo ""
	@echo "=== Development logs ==="
	-docker-compose -f docker/docker-compose.dev.yml logs 2>/dev/null || echo "No development container running"
