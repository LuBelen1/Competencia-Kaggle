import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV,StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
from category_encoders import TargetEncoder

## Modelos que perdieron
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from lightgbm import LGBMClassifier


df = pd.read_csv('C:/Users/HP/Downloads/Predictivo/TP/train_output.csv')
df_test = pd.read_csv('C:/Users/HP/Downloads/Predictivo/TP/test_output.csv')

## EVALUAMOS EL DF
print(df.describe())
print(df.info())
print(df.shape)

## MISSINGS
def missings(df):
    missing_data = df.isnull().sum()
    total_missing = missing_data.sum()
    print(f"\nTotal de valores faltantes en el dataset: {total_missing}")
miss=missings(df)

# Clases desbalanceadas? 
clases=df['condition'].value_counts()
print(f'Cantidad de valores de cada clase (new,used): {clases}') # Hay mas productos nuevos que usados, pero no hay un desbalance notable entre las clases

# Separo columnas para tratar por separado
def separar_cols(df):  
    # Separamos las columnas por tipo de dato
    numericas= df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categoricas= df.select_dtypes(include=['object']).columns.tolist()
    return numericas, categoricas

numericas,categoricas=separar_cols(df)
print(numericas)
df_num=df[numericas]
df_cat=df[categoricas]

## GRAFICOS ##

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Distribución de las Numericas')

sns.histplot(copia['base_price'], kde=True, bins=20, color='lightblue', ax=axes[0, 0])
axes[0, 0].set_title('Base Price')

sns.histplot(copia['price'], kde=True, bins=20, color='lightgreen', ax=axes[0, 1])
axes[0, 1].set_title('Price')

sns.histplot(copia['official_store_id'], kde=True, bins=20, color='lightyellow', ax=axes[1, 0])
axes[1, 0].set_title('Official Store ID')

sns.histplot(copia['initial_quantity'], kde=True, bins=20, color='blue', ax=axes[1, 1])
axes[1, 1].set_title('Initial Quantity')

sns.histplot(copia['sold_quantity'], kde=True, bins=20, color='purple', ax=axes[1, 1])
axes[1, 1].set_title('Sold Quantity')

sns.histplot(copia['available_quantity'], kde=True, bins=20, color='red', ax=axes[1, 1])
axes[1, 1].set_title('Available Quantity')

plt.tight_layout()
plt.show()

# Graficamos las correlaciones
corr_mt = df[numericas].corr()
sns.heatmap(corr_mt,annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.show()

## COLUMNAS CON MISSINGS
def encontrar_columnas(df):
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            print('Columnas con missings:')
            print(col)

# SEPARAMOS MISSINGS DE CATEGORICAS Y NUMERICAS
faltantes_num = encontrar_columnas(df[numericas])
faltantes_cat = encontrar_columnas(df_cat)

# Porcentaje de missings de numericas
missings_num = df_num.isnull().mean() * 100
print(f'Porcentaje de missings en numericas: {missings_num}')

# Porcentaje de missings de categoricas
missings_cat = df_cat.isnull().mean() * 100
print(f'Porcentaje de missings en categoricas: {missings_cat}')

## VALORES UNICOS DE CADA CATEGORIA
print(df['warranty'].value_counts())
print(df['seller_address_city_id'].value_counts())
print(df['seller_address_state_id'].value_counts())
print(df['seller_address_country_id'].value_counts())

### IMPUTACION, DROPEO Y ESTANDARIZACION ###
def limpiar_dataset(df,columnas_a_dropear=None,columnas_num=None,imputaciones=None,columnas_pago=None):   
    if imputaciones:
        for col, value in imputaciones.items():
            df[col].fillna(value, inplace=True)
        
    # 'has_warranty' como binaria (1 si tiene garantía, 0 si no tiene)
    df['has_warranty'] = np.where(df['warranty'].notnull(), 1, 0)

    # 'has_sub_status' para indicar si el producto tiene un 'sub_status' (1 si tiene, 0 si no tiene)
    df['has_sub_status'] = np.where(df['sub_status'].notnull(), 1, 0)

    # 'has_deal' para indicar si el producto es parte de alguna 'deal' (1 si tiene, 0 si no tiene)
    df['has_deal'] = np.where(df['deal_ids'].notnull(), 1, 0)
    
    df['has_alternative_payment'] = df[columnas_pago].any(axis=1).astype(int)
    
    df['price_margin'] = df['price'] - df['base_price']
    
    df['price_ratio_base'] = df['price'] / df['base_price']
    
    df['price_ratio_base'].replace([np.inf, -np.inf], np.nan, inplace=True)
    
    
    if columnas_a_dropear:
        df = df.drop(columns=columnas_a_dropear)
    
    for col in columnas_pago:
        df = df.drop(columns=col)
     
    return df

payment_columns = [
    'non_mercado_pago_payment_methods_1_MLAAM', 'non_mercado_pago_payment_methods_2_MLABC',
    'non_mercado_pago_payment_methods_3_MLACD', 'non_mercado_pago_payment_methods_4_MLADC',
    'non_mercado_pago_payment_methods_5_MLAMC', 'non_mercado_pago_payment_methods_6_MLAMO',
    'non_mercado_pago_payment_methods_7_MLAMP', 'non_mercado_pago_payment_methods_8_MLAMS',
    'non_mercado_pago_payment_methods_9_MLAOT', 'non_mercado_pago_payment_methods_10_MLATB',
    'non_mercado_pago_payment_methods_11_MLAVE', 'non_mercado_pago_payment_methods_12_MLAVS',
    'non_mercado_pago_payment_methods_13_MLAWC', 'non_mercado_pago_payment_methods_14_MLAWT'
]


columnas_num = ['base_price', 'price', 'initial_quantity', 'sold_quantity', 'available_quantity']
imputaciones = {'seller_address_city_id': 'Desconocido','seller_address_country_id': 'AR','seller_address_state_id': 'Desconocido','official_store_id':-1}
enc = ['seller_address_country_id', 'seller_address_state_id', 'seller_address_city_id', 'shipping_mode', 'listing_type_id', 'buying_mode', 'category_id', 'currency_id','status']
columnas_dr = ['shipping_tags','sub_status','deal_ids','warranty'] 

# Llamada a la función y limpieza de los datos
df_limpio = limpiar_dataset(df,columnas_dr,columnas_num,imputaciones,payment_columns)
df_test_limpio = limpiar_dataset(df_test,columnas_dr,columnas_num,imputaciones,payment_columns)

###################################### MODELO GANADOR #####################################

#------------------------------------- XGBOOST -------------------------------#
from scipy import stats
# Encodeamos la y 
y = df_limpio['condition']
X = df_limpio.drop('condition',axis=1)
le = LabelEncoder()
y_en = le.fit_transform(y)  # 'new'== 1, 'used'== 0
X_train,X_test,y_train,y_test = train_test_split(X,y_en,test_size=0.3,random_state=42)

target_encoder = TargetEncoder()
scaler = MinMaxScaler()
transformer = ColumnTransformer(
        transformers=[
            ('target_encoder', target_encoder, enc),
            ('scaler',scaler,columnas_num)
        ],remainder='passthrough')

param_dist = {'model__n_estimators': stats.randint(100, 1000),
              'model__learning_rate': stats.uniform(0.01, 0.99),
              'model__subsample': stats.uniform(0.1, 1),
              'model__max_depth': [3, 4, 5, 6, 7, 8, 9],
              'model__colsample_bytree': stats.uniform(0.1, 1),
              'model__min_child_weight': [1, 2, 3, 4]
             }
classifier = XGBClassifier(random_state=42)

model_clf = Pipeline(steps=[("pre-processor", transformer),
                      ("model", classifier)])
cv = StratifiedKFold(n_splits=3, random_state=2097, shuffle=True)

xgb_cv = RandomizedSearchCV(model_clf,param_dist,cv=cv, verbose=1,scoring='accuracy',n_iter=2,random_state=209)

xgb_fit = xgb_cv.fit(X_train,y_train)

print(f'Accuracy CV: {round(xgb_cv.best_score_,2)}')
print(accuracy_score(xgb_cv.predict(X_test),y_test))

copia = df_test_limpio
X_test_final = df_test_limpio
X_train, X_test_final = X_train.align(X_test_final, join='left', axis=1, fill_value=0) ## Alineamos (si no incluia esta opcion se crasheaba)
y_pred_test = xgb_cv.best_estimator_.predict(X_test_final)

# DataFrame con las predicciones e IDs
predictions_df = pd.DataFrame({
    "ID": range(1, len(y_pred_test) + 1),
    "Predicted_Condition": le.inverse_transform(y_pred_test)  # Revertir el encoding de 'condition'
})

#Guardar en un CSV
predictions_df.to_csv("predicciones_xgb.csv", index=False)
print("Predicciones guardadas en 'predicciones_xgb.csv'")

# ##############################################################################################
# ## CHEQUEO LAS COLUMNAS (use esta funcion para ver que columnas estaban y cuales no en X_train y X_test porque me crasheaba)
# # def verificar_columnas(X_train, X_test):
# #     # Convertimos los nombres de las columnas a conjuntos
# #     columnas_train = set(X_train.columns)
# #     columnas_test = set(X_test.columns)

# #     # Identificar columnas faltantes en cada DataFrame
# #     faltantes_en_test = columnas_train - columnas_test
# #     faltantes_en_train = columnas_test - columnas_train

# #     if not faltantes_en_test and not faltantes_en_train:
# #         print("Las columnas de X_train y X_test coinciden.")
# #     else:
# #         print("Las columnas no coinciden.")
# #verificar_columnas(X_train, X_test_final)
# ############################################################################################## 






