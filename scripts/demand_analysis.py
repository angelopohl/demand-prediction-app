import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import json
import io
import base64

def analyze_demand_data(csv_data):
    """
    Analiza los datos de demanda y retorna resultados del modelo ML
    """
    try:
        # Leer datos CSV
        df = pd.read_csv(io.StringIO(csv_data))
        
        # Copia para trabajar
        df_clean = df.copy()
        
        # Normalizar la columna Order_Demand
        df_clean['Order_Demand'] = df_clean['Order_Demand'].astype(str).str.replace('(', '-').str.replace(')', '')
        df_clean['Order_Demand'] = df_clean['Order_Demand'].astype(float)
        
        # Label Encoding para columnas categóricas
        le_warehouse = LabelEncoder()
        le_category = LabelEncoder()
        le_season = LabelEncoder()
        
        df_clean['Warehouse_Name'] = le_warehouse.fit_transform(df_clean['Warehouse'])
        df_clean['Category_Name'] = le_category.fit_transform(df_clean['Product_Category'])
        df_clean['Season_Code'] = le_season.fit_transform(df_clean['Season'])
        
        # Selección de características y variable objetivo
        X = df_clean[['Warehouse_Name', 'Category_Name', 'Year', 'Month', 'Day', 'Weekday', 'Season_Code', 'Promo']]
        y = df_clean['Order_Demand']
        
        # División de datos
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Modelos a probar
        modelos = {
            'LinearRegression': LinearRegression(),
            'DecisionTree': DecisionTreeRegressor(random_state=42),
            'RandomForest': RandomForestRegressor(random_state=42)
        }
        
        resultados = {}
        
        # Entrenar y evaluar modelos
        for nombre, modelo in modelos.items():
            modelo.fit(X_train, y_train)
            y_pred = modelo.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            resultados[nombre] = {'MSE': float(mse), 'R2': float(r2)}
        
        # Seleccionar mejor modelo (Random Forest)
        rf = RandomForestRegressor(random_state=42)
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)
        
        # Importancia de características
        importances = rf.feature_importances_
        features = X.columns
        feature_importance = [
            {'feature': feature, 'importance': float(importance)}
            for feature, importance in zip(features, importances)
        ]
        feature_importance.sort(key=lambda x: x['importance'], reverse=True)
        
        # Análisis por categoría
        X_test_with_category = X_test.copy()
        X_test_with_category['Product_Category'] = df.loc[X_test.index, 'Product_Category']
        
        result_df = X_test_with_category.copy()
        result_df["Real"] = y_test
        result_df["Predicción"] = rf_pred
        result_df["Error"] = abs(result_df["Real"] - result_df["Predicción"])
        
        # Demanda predicha promedio por categoría
        category_analysis = []
        for category in result_df['Product_Category'].unique():
            category_data = result_df[result_df['Product_Category'] == category]
            avg_demand = float(category_data['Predicción'].mean())
            avg_error = float(category_data['Error'].mean())
            category_analysis.append({
                'category': category,
                'avgDemand': avg_demand,
                'error': avg_error
            })
        
        category_analysis.sort(key=lambda x: x['avgDemand'], reverse=True)
        
        # Determinar mejor modelo
        best_model = max(resultados.keys(), key=lambda k: resultados[k]['R2'])
        
        return {
            'success': True,
            'modelComparison': resultados,
            'bestModel': best_model,
            'featureImportance': feature_importance,
            'categoryAnalysis': category_analysis
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def generate_visualizations(results):
    """
    Genera visualizaciones para el análisis
    """
    try:
        # Configurar estilo
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Crear gráficos
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Comparación de modelos
        models = list(results['modelComparison'].keys())
        r2_scores = [results['modelComparison'][model]['R2'] for model in models]
        
        axes[0, 0].bar(models, r2_scores, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        axes[0, 0].set_title('Comparación de Modelos - R² Score')
        axes[0, 0].set_ylabel('R² Score')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Importancia de características
        features = [item['feature'] for item in results['featureImportance'][:5]]
        importances = [item['importance'] for item in results['featureImportance'][:5]]
        
        axes[0, 1].barh(features, importances, color='#96CEB4')
        axes[0, 1].set_title('Top 5 - Importancia de Características')
        axes[0, 1].set_xlabel('Importancia')
        
        # 3. Demanda por categoría
        categories = [item['category'] for item in results['categoryAnalysis']]
        demands = [item['avgDemand'] for item in results['categoryAnalysis']]
        
        axes[1, 0].bar(categories, demands, color='#FFEAA7')
        axes[1, 0].set_title('Demanda Promedio por Categoría')
        axes[1, 0].set_ylabel('Demanda Promedio')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Error por categoría
        errors = [item['error'] for item in results['categoryAnalysis']]
        
        axes[1, 1].bar(categories, errors, color='#FD79A8')
        axes[1, 1].set_title('Error Promedio por Categoría')
        axes[1, 1].set_ylabel('Error Promedio')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Convertir a base64 para enviar al frontend
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return image_base64
        
    except Exception as e:
        print(f"Error generando visualizaciones: {e}")
        return None

# Ejemplo de uso
if __name__ == "__main__":
    # Datos de ejemplo para prueba
    sample_data = """Warehouse,Product_Category,Year,Month,Day,Weekday,Season,Promo,Order_Demand
    Warehouse_A,Electronics,2023,1,15,1,Winter,1,1200
    Warehouse_B,Clothing,2023,2,20,3,Winter,0,800
    Warehouse_A,Home & Garden,2023,3,10,5,Spring,1,950
    Warehouse_C,Sports,2023,4,5,2,Spring,0,600"""
    
    results = analyze_demand_data(sample_data)
    print(json.dumps(results, indent=2))
