import joblib
import pandas as pd
import warnings
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
import sklearn
# Usamos un print simple para verificar la versión
print(sklearn.__version__) 
warnings.simplefilter('ignore')

import logging
from sys import stdout

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logFormatter = logging.Formatter("%(asctime)s %(levelname)s %(filename)s: %(message)s")
consoleHandler = logging.StreamHandler(stdout)
consoleHandler.setFormatter(logFormatter)
logger.addHandler(consoleHandler)

# =======================================================================
# ⚠️ CLASES CUSTOMIZADAS DEL PIPELINE (NECESARIAS PARA joblib.load) ⚠️
# Las clases deben estar definidas en el script de inferencia
# para que joblib pueda recrear el objeto Pipeline correctamente.
# =======================================================================

# Clase para imputación global final de respaldo 
class GlobalImputer(BaseEstimator, TransformerMixin):
    """
    Imputa NaNs en todo el DataFrame: 
    - Variables categóricas: por 'missing'.
    - Variables numéricas: por mediana.
    """
    def fit(self, X, y=None):
        self.median_imputer = SimpleImputer(strategy='median')
        self.mode_imputer = SimpleImputer(strategy='constant', fill_value='missing')
        
        num_cols = X.select_dtypes(exclude='object').columns
        cat_cols = X.select_dtypes(include='object').columns
        
        if len(num_cols) > 0:
            self.median_imputer.fit(X[num_cols])
        if len(cat_cols) > 0:
            self.mode_imputer.fit(X[cat_cols])
            
        return self

    def transform(self, X):
        X_copy = X.copy()
        num_cols = X_copy.select_dtypes(exclude='object').columns
        cat_cols = X_copy.select_dtypes(include='object').columns
        
        if len(num_cols) > 0:
            X_copy[num_cols] = self.median_imputer.transform(X_copy[num_cols])
        if len(cat_cols) > 0:
            X_copy[cat_cols] = self.mode_imputer.transform(X_copy[cat_cols])
            
        return X_copy


# Clases simuladas para el pipeline. 
class FeaturesEngineering(BaseEstimator, TransformerMixin):
    def __init__(self, map_localidad_region=None): self.map_localidad_region = map_localidad_region
    def fit(self, X, y=None): return self
    def transform(self, X): return X
    
class StratifiedOutlierRemover(BaseEstimator, TransformerMixin):
    def __init__(self, methods_dict=None): self.methods_dict = methods_dict
    def fit(self, X, y=None): return self
    def transform(self, X): return X

class MissingByPercentageImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X): return X

class ModeImputer(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None): self.columns = columns
    def fit(self, X, y=None): return self
    def transform(self, X): return X

# =======================================================================

# Carga del modelo serializado
pipeline = joblib.load('pipeline.pkl')

logger.info('loaded pipeline')

# Lee el archivo de entrada desde la ruta del volumen montado por Docker
df_input = pd.read_csv('/files/input.csv')

logger.info('loaded input')

print("--- Primeras 5 filas del input para inferencia ---")
print(df_input.head())
print("--------------------------------------------------")

# ⚠️ PASO CRÍTICO: ELIMINAR COLUMNAS NO ESPERADAS POR EL PIPELINE ⚠️
# 'Date' es metadata, y 'RainTomorrow' es la etiqueta. Ambas rompen la inferencia.
cols_to_drop = ['Date']

# Aseguramos que RainTomorrow se elimine solo si existe (para evitar error si el usuario ya lo quitó)
if 'RainTomorrow' in df_input.columns:
    cols_to_drop.append('RainTomorrow')

# Eliminamos las columnas antes de pasar a la predicción
df_features = df_input.drop(columns=cols_to_drop, errors='ignore')

logger.info(f"dropped columns: {cols_to_drop}")

# Realiza la predicción sobre el set de features limpio
output = pipeline.predict(df_features)

logger.info('made predictions')

# Creamos el DataFrame de salida y lo guardamos
df_output = df_input.copy()
df_output['RainTomorrow_predicted'] = output

df_output.to_csv('/files/output.csv', index=False)

logger.info('saved output')