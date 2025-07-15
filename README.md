# Pronóstico de Demanda Energética con LSTM

Este proyecto tiene como objetivo predecir la demanda diaria de energía eléctrica en la ciudad de **Cali, Colombia**, utilizando un modelo de red neuronal recurrente de tipo **LSTM (Long Short-Term Memory)**. 

Los datos históricos se obtienen de la API pública de **XM**, operador del Sistema Interconectado Nacional y administrador del Mercado de Energía Mayorista de Colombia.

El flujo completo abarca:

- Extracción de datos desde la API de XM
- Preprocesamiento y creación de características
- Entrenamiento y evaluación del modelo
- Visualización de resultados a través de un **dashboard interactivo**

---

## Estructura del Proyecto

```
Pronostico_De_Demanda/
│
├── data/
│   └── xm_api_data.csv                  # Datos brutos descargados de la API de XM
│
├── notebooks/
│   └── Pronostico_Demanda_Energetica_LSTM.ipynb  # Proceso completo de experimentación
│
├── src/
│   ├── data_loader.py                  # Script para descargar los datos
│   ├── processing.py                   # Funciones de limpieza y preprocesamiento
│   ├── lstm_model.py                   # Arquitectura del modelo LSTM
│   ├── train.py                        # Entrenamiento del modelo
│   ├── evaluate.py                     # Evaluación del modelo
│   └── utils.py                        # Funciones auxiliares
│
├── dashboard/
│   └── app.py                          # Dashboard interactivo con Streamlit
│
├── results/
│   ├── predicciones_vs_reales.png     # Gráfico de resultados
│   └── metrics.txt                    # Métricas (RMSE, MAE, R²)
│
├── venv/                               # Entorno virtual (opcional)
├── requirements.txt                    # Dependencias del proyecto
└── README.md
```

---

## Cómo Empezar

### Requisitos Previos

- Python 3.8 o superior
- Git
- Jupyter Notebook
- (Opcional) Virtualenv o conda

### Instalación

```bash
git clone <URL-del-repositorio>
cd Pronostico_De_Demanda

# Crear y activar entorno virtual
python -m venv venv
# Windows
.env\Scripts\ctivate
# macOS/Linux
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

---

## Uso del Proyecto

### 1. Obtener los datos

Puedes descargar los datos desde la API de XM:

- Usando el notebook: `notebooks/Pronostico_Demanda_Energetica_LSTM.ipynb`
- O ejecutando directamente:

```bash
python src/data_loader.py
```

### 2. Entrenamiento del Modelo

Sigue los pasos del notebook:

- Limpieza de datos
- Ingeniería de características (lags, festivos, etc.)
- Escalado
- Entrenamiento y evaluación del modelo
- Guardado de resultados

### 3. Dashboard Interactivo

Ejecuta la app para visualizar los resultados:

```bash
streamlit run dashboard/app.py
```

Podrás:

- Ver predicciones vs valores reales
- Obtener una predicción para un día específico
- Explorar métricas del modelo

---

## Resultados

Después del entrenamiento, en la carpeta `results/` encontrarás:

- `metrics.txt`: Métricas como RMSE, MAE y R².
- `predicciones_lstm`: Comparación gráfica entre la demanda real y la predicha.

---

## Dependencias Principales

- `pandas`: Manipulación de datos
- `numpy`: Cálculos numéricos
- `torch`: Red neuronal LSTM
- `scikit-learn`: Escalado y métricas
- `streamlit`: Dashboard interactivo
- `plotly`: Gráficas interactivas
- `holidays`: Cálculo de días festivos en Colombia

---

## Autor

**Miguel Correa**  
Ingeniería en Energía Inteligente – Universidad Icesi  
GitHub: [@miguelcorrea](https://github.com/miguelcorrea)

---

## Licencia

Este proyecto está bajo la licencia MIT. Consulta el archivo [LICENSE](LICENSE) para más información.