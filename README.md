## 🎯 Objetivo del Proyecto
Construir un modelo para predecir qué personas han dejado de fumar hace más de un año, basado en características demográficas, psicológicas y hábitos previos de consumo.

## 📦 Estructura del Proyecto
- `notebooks/`: notebooks divididos por etapa (procesamiento, exploración, entrenamiento, evaluación)
- `src/`: funciones auxiliares reutilizables
- `models/`: modelos entrenados y umbrales
- `outputs/`: métricas, predicciones y visualizaciones

## 🧠 Modelos Entrenados
- LightGBM (Optuna + SMOTE)
- XGBoost (Optuna + SMOTE)
- Stacking & Voting Ensemble

## ⚖️ Métricas
- Métrica objetivo: F2-score (recall prioritario)
- ROC AUC promedio > 0.76
