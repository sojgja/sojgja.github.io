test:
	@python tests.py

me:
	cat Makefile

git:
	git add .
	git commit -m "add coding"
	git push

model:
	python manage.py makemigrations
	python manage.py makemigrations www
	python manage.py migrate
	python manage.py migrate www

build:
	docker build -t soigia.django .

run docker:
	docker-compose down -v
	docker-compose up -d --build
	docker ps -a

cli:
	docker-compose run --rm cli python cli.py

server:
	docker-compose run --rm www python manage.py runserver 8001

code:
	docker exec -it cli /bin/bash

shell:
	docker-compose run --rm www python manage.py shell

setup i:
	pip install -e .

demo:
	soigia --name "World"

load:
	python manage.py load_csv soigia/cs.csv

clone:
	echo git clone git@github.com:squidfunk/mkdocs-material.git