import joblib
import pandas as pd
import warnings
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
import sklearn
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
# Estas definiciones DEBEN COINCIDIR EXACTAMENTE con las usadas en train.py
# =======================================================================

# Clase para imputación global final de respaldo (para atrapar NaNs no cubiertos)
class GlobalImputer(BaseEstimator, TransformerMixin):
    """
    Imputa NaNs en todo el DataFrame: 
    - Variables categóricas: por 'missing' (o moda si se prefiere).
    - Variables numéricas: por mediana.
    """
    def fit(self, X, y=None):
        self.median_imputer = SimpleImputer(strategy='median')
        self.mode_imputer = SimpleImputer(strategy='constant', fill_value='missing')
        
        # Fit solo en columnas que contienen los tipos.
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
        
        # Aplicar imputación
        if len(num_cols) > 0:
            X_copy[num_cols] = self.median_imputer.transform(X_copy[num_cols])
        if len(cat_cols) > 0:
            X_copy[cat_cols] = self.mode_imputer.transform(X_copy[cat_cols])
            
        return X_copy


# Ejemplos de clases simuladas usadas en el pipeline de entrenamiento.
# ¡REEMPLAZA CON TUS CLASES REALES si difieren de estas simulaciones!
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

pipeline = joblib.load('pipeline.pkl')

logger.info('loaded pipeline')

# La ruta '/files/input.csv' es la ruta esperada dentro del contenedor Docker (ver Dockerfile)
df_input = pd.read_csv('/files/input.csv')

logger.info('loaded input')

print("--- Primeras 5 filas del input para inferencia ---")
print(df_input.head())
print("--------------------------------------------------")

# Nota: El pipeline fue entrenado con la columna 'Date' eliminada en X_train,
# y la columna objetivo 'RainTomorrow' eliminada. df_input debe tener
# el mismo conjunto de columnas de características.

output = pipeline.predict(df_input)

logger.info('made predictions')

# El target original era 'RainTomorrow' (binario 1/0). 
# Adaptamos el nombre de la columna de salida a 'RainTomorrow_predicted'.
# La ruta '/files/output.csv' es donde el contenedor guardará las predicciones.
pd.DataFrame(output, columns=['RainTomorrow_predicted']).to_csv('/files/output.csv', index=False)

logger.info('saved output')