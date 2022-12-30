# VART AI

Este repositorio contiene el código fuente utilizado en el desarrollo del módulo de inteligencia artificial para el sistema VART (Valoración de Retina por Telemedicina). Este sistema de inteligencia artificial indica la salud de la retina de bebés prematuros a partir de una imagen entregada.

## Motivación
La segmentación automática de imágenes médicas es un proceso que permite la extracción de información útil que apoya los procesos de diagnóstico de los médicos, sobretodo cuando se tratan patologías de alta complejidad que solo puede ser diagnosticada por pocos especialistas. En esta implementación, este proceso se utiliza para segmentar los vasos de la retina con el fin de medir la propagación de estos a lo largo de la superficie de la retina en bebés prematuros. Esto ayuda a identificar posibles casos de retinopatía en prematuros de manera ágil y precisa.

## Instalación de requerimientos y dependencias
Para este punto se asume que el servidor o equipo a utilizar cuenta con una instalación activa de Python en su versión 3.8 o superior. Además del gestor de paquetes de Python llamado pip para la instalación de los recursos y librerías necesarias. Adicionalmente se debe clonar este repositorio en el lugar que se desee dentro del equipo o servidor.

```
git clone https://github.com/ufotechco/vart-ai.git
```

Al clonar el proyecto se presentan los siguientes archivos.

- **VartYolo**: Carpeta relacionada al módulo para detección de estructuras a partir de su entrenamiento.
- **VesselDetection**: Carpeta relacionada al módulo para detección de vasos sanguíneos.
- **veins_detector**: Carpeta que integra distintos elementos del sistema.

### Instalación de dependencias
Para la instalación de dependencias se ejecuta lo siguiente.

1. Ingresamos a la carpeta VartYolo y ejecutamos el siguiente comando
```
pip install -r requirements.txt
```

2. Luego ingresamos a la carpeta VesselDetection y ejecutamos el siguiente comando
```
pip install -r requirements.txt
```

## Uso del sistema
Para utilizar el sistema de inteligencia artificial debemos ingresar a la carpeta **veins_detector** ejecutar el siguiente comando.
```
python veins_detector.py --source dir_path
```

Este comando requiere del siguiente argumento: 
- **source**: Corresponde a **la ruta donde se encuentran las imágenes que serán procesadas por el sistema**.

El procedimiento se lleva a cabo a través de las siguientes etapas, pero el resultado final consiste en la respuesta del sistema indicando si la imagen entregada corresponde a un ojo sano o enfermo.

![Proyecto nuevo (4)](https://user-images.githubusercontent.com/5098241/210098728-12765a85-819e-4f5e-b12b-0a38f5b5ffbd.png)

## Requerimientos del sistema
Para la ejecución del sistema de AI se requieren las siguientes características mínimas del servidor o equipo donde será ejecutado.
- 4GB de memoria RAM
- Procesador Intel Xenon 2 núcleos
- Disco duro con mínimo 50GB de almacenamiento de estado sólido (SSD o NVMe SSD)

