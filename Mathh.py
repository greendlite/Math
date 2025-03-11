import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, f1_score, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler , OneHotEncoder
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

table = pd.read_excel('dataset.xlsx', sheet_name='online_sales_dataset')#Загрузка данных
cat_col = ['InvoiceNo', 'StockCode', 'Description', 'CustomerID', 'Country', 
           'PaymentMethod', 'Category', 'SalesChannel', 'ReturnStatus', 
           'ShipmentProvider', 'WarehouseLocation', 'OrderPriority']

onehot = pd.get_dummies(table[cat_col], drop_first=True) # Преобразование категориальных переменных
table_proces = pd.concat([table.drop(columns=cat_col), onehot], axis=1) # Объединение данных
table_proces['SalesAmount'] =  table_proces['Quantity'] * ((table_proces['UnitPrice'] * 1 - table_proces['Discount']) + table_proces['ShippingCost'])
print(table_proces)
# 1. .......................................................Регрессия.......................................................
print("\n1. Регрессия:")
y_reg = table_proces['SalesAmount']
X_reg = table_proces[['ShipmentProvider_UPS', 'ShipmentProvider_Royal Mail', 'ShipmentProvider_FedEx','WarehouseLocation_Berlin','WarehouseLocation_London','WarehouseLocation_Paris','WarehouseLocation_Rome']]

X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=123)# разделение на случайные множества
regressor = LinearRegression() #Построение линейной регрессии
regressor.fit(X_train, y_train) # Машинное обучение для регрессии.
y_pred = regressor.predict(X_test)# Предсказание меток данных на основе обученной модели

plt.scatter(x=y_test, y=y_pred, alpha=0.3, color='red')
plt.ylabel('Предсказанные')
plt.xlabel('Фактические')
plt.title('Сравнение фактических и предсказанных значений')
plt.show()

print(f"Среднеквадратичная ошибка (MES): {mean_squared_error(y_test, y_pred):.2f}")
print(f"Коэффициент детерминации (R^2): {r2_score(y_test, y_pred):.4f}")
# .......................................................Кластеризация методом ближайших соседей ....   ...................................................
print("\n2.Кластеризация:")

numerical_cols = ['Quantity', 'UnitPrice', 'Discount']
categorical_cols = ['Description', 'Category', 'SalesChannel']

# 3. Предобработка данных
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
    ]
)

X_processed = preprocessor.fit_transform(table_proces)

# 4. Кластеризация
# Метод локтя для определения оптимального числа кластеров
inertia = []
k_range = range(1, 10)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_processed)
    inertia.append(kmeans.inertia_)

# Построение графика метода локтя
plt.figure(figsize=(8, 5))
plt.plot(k_range, inertia, marker="o")
plt.title("Метод Локтя", fontsize=16)
plt.xlabel("K", fontsize=14)
plt.ylabel("SSE", fontsize=14)
plt.xticks(k_range)
plt.grid(True)
plt.show()

# Выбор оптимального числа кластеров
optimal_k = 3 # Заменить на оптимальное значение, найденное методом локтя
kmeans = KMeans(n_clusters=optimal_k, random_state=123)
clusters = kmeans.fit_predict(X_processed)

# 5. Добавление кластеров в DataFrame
table_proces["Cluster"] = clusters

# 6. Визуализация результатов
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=table['UnitPrice'], 
    y=table['Quantity'], 
    hue=table_proces['Cluster'], 
    palette="viridis", 
    s=100
)
plt.title("K-means Clustering of Products", fontsize=16)
plt.xlabel("UnitPrice", fontsize=14)
plt.ylabel("Quantity", fontsize=14)
plt.legend(title="Cluster", fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()

# Расчет средней оценки силуэта
silhouette_avg = silhouette_score(X_processed, clusters)
print(f"Average Silhouette Score: {silhouette_avg:.2f}")

# 7. Анализ кластеров
for i in range(optimal_k):
    print(f"\nCluster {i}:")
    cluster_data = table_proces[table_proces['Cluster'] == i]
    print(cluster_data[['StockCode', 'Quantity', 'UnitPrice', 'Discount']].head())
cluster_counts = table_proces['Cluster'].value_counts()
print("Количество товаров в каждом кластере:")
print(cluster_counts)

# .......................................................Классификация методом k-ближайших соседей.......................................................
print("\n--- Классификация ---")
X_class = table_proces.drop(columns=['OrderPriority_Low', 'OrderPriority_Medium'])
y_class = table['OrderPriority']
print("\nРаспределение классов:")
print(y_class.value_counts())
smote = SMOTE(random_state=42)#Использование алгоритма для удаления дисбаланса
X_class_resampled, y_class_resampled = smote.fit_resample(X_class, y_class) #Алгоритм избыточной выборки меньшинства
scaler_classification = StandardScaler()# Стандартизирование данных
X_class_resampled_scaled = scaler_classification.fit_transform(X_class_resampled) # Маштабирование данных
X_train, X_test, y_train, y_test = train_test_split(X_class_resampled_scaled, y_class_resampled, test_size=0.2, random_state=42) # Разделение данных на обучающую и тестовую выборки

# Обучение классификатора K-Nearest Neighbors
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train, y_train)

# Оценка модели
y_pred_class = knn_classifier.predict(X_test)
accuracy = (y_pred_class == y_test).mean()
f1 = f1_score(y_test, y_pred_class, average='macro')

print(f"Точность классификации (kNN): {accuracy:.2f}")
print(f"F1-Score (kNN): {f1:.2f}")
report = classification_report(y_test, y_pred_class, output_dict=True)

# Форматированный вывод метрик
for class_label, metrics in report.items():
    if class_label == 'accuracy':
        continue  # Пропускаем строку с точностью
    print(f" Класс {class_label}: precision = {metrics['precision']:.2f}, recall = {metrics['recall']:.2f}, f1-score = {metrics['f1-score']:.2f}, support = {metrics['support']:.0f}")
conf_matrix = confusion_matrix(y_test, y_pred_class)
print("\nМатрица путаницы:")
print(conf_matrix)