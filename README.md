# Image to 3D Converter

Una aplicación web que convierte imágenes 2D en modelos 3D utilizando TensorFlow.js y Three.js.

## Instalación

1. Clona este repositorio:
```bash
git clone https://github.com/brisagiuliana/3d.git
cd 3d
```

2. Combina los archivos del modelo:
```bash
./scripts/combine-model.sh
```

3. Inicia un servidor web local:
```bash
python3 -m http.server 8000
```

4. Abre tu navegador y visita:
```
http://localhost:8000
```

## Uso

1. Haz clic en el botón "Seleccionar archivo" para cargar una imagen
2. Espera a que el modelo procese la imagen
3. El modelo 3D generado se mostrará en la pantalla
4. Puedes rotar y hacer zoom en el modelo usando el mouse

## Tecnologías utilizadas

- TensorFlow.js
- Three.js
- HTML/CSS/JavaScript
