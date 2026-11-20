# Proyecto Final - Hiper-heurística de Selección para TSP

## Descripción

Este proyecto implementa una hiper-heurística de selección para el Problema del Agente Viajero (TSP).

El objetivo consiste en predecir cuál de los siguientes algoritmos obtiene la mejor solución para una instancia dada:

- 2-opt Hill Climbing
- Simulated Annealing + 2-opt
- MST + 2-opt

Para ello se generó un conjunto de datos etiquetado ejecutando los tres algoritmos sobre las instancias proporcionadas y posteriormente se entrenó un modelo de Machine Learning para predecir el algoritmo ganador.

---

## Estructura del proyecto

```text
ProyectoFinal/
│
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── instances/
│
├── label_instances.py
├── train_model.py
│
├── train_labeled.csv
├── test_labeled.csv
│
├── extras/
│
└── README.md
