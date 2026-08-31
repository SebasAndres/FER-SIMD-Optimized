docker compose build
xhost +local:docker
docker compose run fer bash -c "cd src && make pipeline"