import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import json
import sys
import warnings
warnings.filterwarnings('ignore')

def analyze_demand_data(csv_path, output_path):
    """
    Analiza los datos de demanda replicando exactamente el código de Google Colab
    """
    try:
        # Leer el archivo CSV
        print(f"Leyendo archivo: {csv_path}")
        df = pd.read_csv(csv_path)
        
        print(f"Datos cargados: {len(df)} filas, {len(df.columns)} columnas")
        print(f"Columnas: {list(df.columns)}")
        
        # Verificar columnas requeridas
        required_columns = ['Warehouse', 'Product_Category', 'Year', 'Month', 'Day', 'Weekday', 'Season', 'Promo', 'Order_Demand']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Faltan las siguientes columnas requeridas: {missing_columns}")
        
        # Copia para trabajar
        df_clean = df.copy()
        
        # Normalizar la columna Order_Demand por si hay strings negativos como '(100)'
        print("Normalizando Order_Demand...")
        df_clean['Order_Demand'] = df_clean['Order_Demand'].astype(str).str.replace('(', '-').str.replace(')', '')
        df_clean['Order_Demand'] = pd.to_numeric(df_clean['Order_Demand'], errors='coerce')
        
        # Eliminar filas con valores NaN en Order_Demand
        df_clean = df_clean.dropna(subset=['Order_Demand'])
        
        print(f"Datos después de limpieza: {len(df_clean)} filas")
        
        # Label Encoding para columnas categóricas
        print("Aplicando Label Encoding...")
        le_warehouse = LabelEncoder()
        le_category = LabelEncoder()
        le_season = LabelEncoder()
        
        df_clean['Warehouse_Name'] = le_warehouse.fit_transform(df_clean['Warehouse'].astype(str))
        df_clean['Category_Name'] = le_category.fit_transform(df_clean['Product_Category'].astype(str))
        df_clean['Season_Code'] = le_season.fit_transform(df_clean['Season'].astype(str))
        
        # Selección de características y variable objetivo
        feature_columns = ['Warehouse_Name', 'Category_Name', 'Year', 'Month', 'Day', 'Weekday', 'Season_Code', 'Promo']
        X = df_clean[feature_columns]
        y = df_clean['Order_Demand']
        
        print(f"Características: {X.shape}, Objetivo: {y.shape}")
        
        # División de datos
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print("Entrenando modelos...")
        
        # Lista de modelos a probar
        modelos = {
            'LinearRegression': LinearRegression(),
            'DecisionTree': DecisionTreeRegressor(random_state=42),
            'RandomForest': RandomForestRegressor(random_state=42, n_estimators=100)
        }
        
        resultados_modelos = {}
        
        # Entrenar y evaluar cada modelo
        for nombre, modelo in modelos.items():
            print(f"Entrenando {nombre}...")
            modelo.fit(X_train, y_train)
            y_pred = modelo.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            resultados_modelos[nombre] = {
                'MSE': float(mse),
                'R2': float(r2)
            }
            
            print(f'{nombre}: MSE={mse:.2f}, R²={r2:.4f}')
        
        # Entrenar Random Forest para análisis detallado
        print("Analizando Random Forest en detalle...")
        rf = RandomForestRegressor(random_state=42, n_estimators=100)
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)
        
        # Importancia de características
        importances = rf.feature_importances_
        features = X.columns
        indices = np.argsort(importances)[::-1]
        
        feature_importance = []
        for i in range(len(features)):
            feature_importance.append({
                'feature': features[indices[i]],
                'importance': float(importances[indices[i]])
            })
        
        # Análisis por categoría
        print("Analizando por categoría...")
        X_test_with_category = X_test.copy()
        X_test_with_category['Product_Category'] = df_clean.loc[X_test.index, 'Product_Category'].values
        
        result_df = X_test_with_category.copy()
        result_df["Real"] = y_test.values
        result_df["Predicción"] = rf_pred
        result_df["Error"] = abs(result_df["Real"] - result_df["Predicción"])
        
        # Calcular estadísticas por categoría
        category_stats = []
        for category in result_df['Product_Category'].unique():
            category_data = result_df[result_df['Product_Category'] == category]
            if len(category_data) > 0:
                avg_demand = float(category_data['Predicción'].mean())
                avg_error = float(category_data['Error'].mean())
                count = len(category_data)
                
                category_stats.append({
                    'category': str(category),
                    'avgDemand': avg_demand,
                    'error': avg_error,
                    'count': count
                })
        
        # Ordenar por demanda promedio
        category_stats.sort(key=lambda x: x['avgDemand'], reverse=True)
        
        # Determinar el mejor modelo
        best_model = max(resultados_modelos.keys(), key=lambda k: resultados_modelos[k]['R2'])
        
        # Estadísticas generales
        total_samples = len(df_clean)
        unique_categories = len(df_clean['Product_Category'].unique())
        unique_warehouses = len(df_clean['Warehouse'].unique())
        date_range = f"{df_clean['Year'].min()}-{df_clean['Year'].max()}"
        
        # Preparar resultados finales
        results = {
            'success': True,
            'modelComparison': resultados_modelos,
            'bestModel': best_model,
            'featureImportance': feature_importance,
            'categoryAnalysis': category_stats,
            'dataInfo': {
                'totalSamples': total_samples,
                'uniqueCategories': unique_categories,
                'uniqueWarehouses': unique_warehouses,
                'dateRange': date_range,
                'avgDemand': float(df_clean['Order_Demand'].mean()),
                'maxDemand': float(df_clean['Order_Demand'].max()),
                'minDemand': float(df_clean['Order_Demand'].min())
            }
        }
        
        # Guardar resultados
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Análisis completado. Resultados guardados en: {output_path}")
        return results
        
    except Exception as e:
        error_result = {
            'success': False,
            'error': str(e)
        }
        
        # Guardar error
        with open(output_path, 'w') as f:
            json.dump(error_result, f, indent=2)
        
        print(f"Error en el análisis: {e}")
        return error_result

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Uso: python demand_analysis.py <csv_path> <output_path>")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    output_path = sys.argv[2]
    
    analyze_demand_data(csv_path, output_path)
