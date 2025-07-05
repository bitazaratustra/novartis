# 🚭 Predicción de Cese del Consumo de Nicotina

## 🎯 Objetivo del Proyecto
Construir un modelo para predecir qué personas han dejado de fumar hace más de un año, basado en características demográficas, psicológicas y hábitos previos de consumo.

## 📦 Estructura del Proyecto
- `data/`: datos crudos y procesados
- `notebooks/`: notebooks divididos por etapa (procesamiento, exploración, entrenamiento, evaluación)
- `src/`: funciones auxiliares reutilizables
- `models/`: modelos entrenados y umbrales

## 🧠 Modelos Entrenados
- LightGBM (Optuna + SMOTE)
- XGBoost (Optuna + SMOTE)
- Stacking & Voting Ensemble

## ⚖️ Métricas
- Métrica objetivo: F2-score (recall prioritario)
- ROC AUC promedio


## 🧠 Modelos Entrenados

Se utilizaron modelos basados en gradient boosting optimizados con **Optuna** y balanceados con **SMOTE**:

- ✅ `LightGBM`
- ✅ `XGBoost`
- ✅ `Stacking Ensemble` (meta-modelo: Regresión Logística)
- ✅ `Voting Ensemble` (ponderación suave entre LGBM y XGB)

---

## ⚖️ Métricas de Evaluación

- **Métrica principal:** `F2-score` (prioriza recall)
- **Otras métricas:** ROC AUC, Precisión, Matriz de confusión

---

## 🔍 Interpretabilidad con SHAP

Se utilizó **SHAP (SHapley Additive exPlanations)** para interpretar el comportamiento de los modelos:

- 📊 **Importancia global**: variables como edad, nivel educativo, impulsividad y patrones de consumo de otras sustancias resultaron clave.

Los resultados de SHAP validaron los hallazgos estadísticos previos (correlaciones, T-tests, clusters).

---

## 📈 Curvas de Aprendizaje

Se generaron **learning curves** para analizar la capacidad generalizadora de los modelos:

- Permiten evaluar **overfitting** o **underfitting**
- Demuestran la eficiencia de los modelos al aumentar la cantidad de datos de entrenamiento
