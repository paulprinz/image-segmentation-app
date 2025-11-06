.PHONY: help build up down logs clean restart status

help:
	@echo "Available commands:"
	@echo "  make build    - Build all Docker images"
	@echo "  make up       - Start all services"
	@echo "  make down     - Stop all services"
	@echo "  make logs     - View logs from all services"
	@echo "  make clean    - Remove all containers, images, and volumes"
	@echo "  make restart  - Restart all services"
	@echo "  make status   - Show status of all services"

build:
	docker-compose build

up:
	docker-compose up -d

down:
	docker-compose down

logs:
	docker-compose logs -f

clean:
	docker-compose down -v
	docker system prune -af

restart:
	docker-compose restart

status:
	docker-compose ps
