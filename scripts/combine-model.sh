#!/bin/bash

# Crear el directorio de assets si no existe
mkdir -p assets

# Combinar las partes del modelo
cat assets/midas.tflite.part_* > assets/midas.tflite

echo "Modelo combinado exitosamente"
