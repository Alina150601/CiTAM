# Презентация
https://www.canva.com/design/DAGm_sMriks/Ns17aH4mLtfMhVeouvSnMg/edit

# Набор данных для задания:
Bank Marketing
https://www.kaggle.com/datasets/henriqueyamahata/bankmarketing

## Код выполнялся в среде - colab.research.google
[https://colab.research.google.com/drive/1s7l1jNSJRb8mqdeSx7Zjr8ydz18f4BH0#scrollTo=sy3vHhvnH5PO](https://colab.research.google.com/drive/1irQVfj_3Lh-v_jdh3KSKgPWwusoAKNgu?usp=sharing) \
При запуске нужно загрузить данные, оставлю файл в этой же папке

```
from google.colab import files
uploaded = files.upload()

import pandas as pd
df = pd.read_csv('bank-additional-full.csv', sep=';')

import matplotlib.pyplot as plt
import seaborn as sns

# Пример: гистограмма возраста
sns.histplot(df['age'], bins=30, kde=True)
plt.title('Возраст клиентов')
plt.xlabel('Возраст')
plt.ylabel('Количество')
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt

target_counts = df['y'].value_counts()

sns.barplot(x=target_counts.index, y=target_counts.values, palette='pastel')
plt.title('Распределение целевой переменной (Подписка на депозит)')
plt.ylabel('Количество')
plt.xlabel('Подписка')
plt.show()

sns.histplot(df['age'], bins=30, kde=True, color='skyblue')
plt.title('Распределение возраста клиентов')
plt.xlabel('Возраст')
plt.ylabel('Количество клиентов')
plt.show()

job_counts = df['job'].value_counts()

sns.barplot(y=job_counts.index, x=job_counts.values, palette='muted')
plt.title('Распределение по профессиям')
plt.xlabel('Количество клиентов')
plt.ylabel('Профессия')
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Целевая переменная
y = df['y'].map({'yes': 1, 'no': 0})

# Удалим duration — его не знают до звонка
X = df.drop(columns=['y', 'duration'])

# Разделим на числовые и категориальные признаки
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

# Создаём предобработчик
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

# Создаём pipeline
clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])

# Делим данные
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучаем модель
clf.fit(X_train, y_train)


# Предсказания
y_pred = clf.predict(X_test)

# Метрики
print("Отчёт по классификации:")
print(classification_report(y_test, y_pred))

# Матрица ошибок
import seaborn as sns
import matplotlib.pyplot as plt
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Предсказание')
plt.ylabel('Истинное значение')
plt.title('Матрица ошибок')
plt.show()
```
