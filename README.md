## Ejecución con Docker

Construir la imagen:

docker build -t proyecto-aves .

Ejecutar el contenedor:

docker run --rm -p 8501:8501 --name aves-app proyecto-aves

Abrir la aplicación en el navegador:

http://localhost:8501
