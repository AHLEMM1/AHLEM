import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_repor
#le chergement des données
diabetes=pd.read_csv('diabetes.csv')
print(diabetes)
class Patient:
    def __init__(self, data_row):
        self.pregnancies = data_row["Pregnancies"]
        self.glucose = data_row["Glucose"]
        self.blood_pressure = data_row["BloodPressure"]
        self.bmi = data_row["BMI"]
        self.age = data_row["Age"]
        self.outcome = data_row["Outcome"]
    def is_diabetic(self):
        return self.outcome == 1

    def __str__(self):
        état_de_santé= "مصاب بالسكري" if self.is_diabetic() else "غير مصاب" 
        return f"العمر: {self.age} |Outcome: {self.outcome} |état de santé : {état_de_santé}"
for index, row in diabetes.iterrows():
    patient = Patient(row)
    print(patient)
#######le manipulation des donnée#######
# Afficher les 5 premières ou les 5 dernières lignes.
print(diabetes.head())
#عرض آخر 5 صفوف
print(diabetes.tail())
#معرفة أسماء الأعمدة (colonnes)
print(diabetes.columns)
#معرفة نوع البيانات لكل عمود (dtypes)
print(diabetes.dtypes)
#غيير نوع بيانات عمود (par astype)
print(diabetes['Glucose'].astype('float64'))
#الوصول إلى أعمدة (Accéder à une variable)
print(diabetes['Glucose'])
#الوصول إلى عدة أعمدة معًا
print(diabetes[['Glucose', 'Insulin']])
#لوصول إلى قيمة معينة (indexing)
print(diabetes['Glucose'][5])
#جزء من العمود (Slicing)
print(diabetes['Glucose'][2:7])

print(diabetes.iloc[3:8, 1:4])
#أصغر قيمة
print(diabetes['Glucose'].min())
#عدد التكرارات لكل قيمة 
print(diabetes['BMI'].sort_values())
# ترتيب القيم
print(diabetes['Pregnancies'].sort_values(ascending=False))
########## le filtrage des donnée##########
print(diabetes.loc[diabetes['Pregnancies'] > 13, :])
print(diabetes.loc[diabetes['Pregnancies'] > 13, ['Glucose', 'BMI', 'Pregnancies']])
print(diabetes.loc[(diabetes['Pregnancies'] > 13) | (diabetes['Glucose'] < 115), ['Pregnancies', 'Glucose', 'Insulin']])
###########le calcul de statistiques descriptives########v
print(diabetes.describe())
###########les valeur is null############
print(diabetes.isnull())
print(diabetes.isnull().sum())
#########Exploration des distributions des données)######v####
print(diabetes['Insulin'].hist())
print(diabetes['Glucose'].plot.kde())
############# Définir les variables explicatives (X) et la variable cible (y)
X = diabetes.drop('Outcome', axis=1)
y = diabetes['Outcome']

# Diviser les données en ensemble d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Créer et entraîner le modèle d'arbre de décision
tree_model = DecisionTreeClassifier(max_depth=4, random_state=42)
tree_model.fit(X_train, y_train)

# Prédictions
y_pred = tree_model.predict(X_test)

# Évaluation du modèle
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Visualisation de l'arbre de décision
plt.figure(figsize=(20,10))
plot_tree(tree_model, feature_names=X.columns, class_names=["Non Diabetic", "Diabetic"], filled=True, rounded=True)
plt.title("Arbre de décision pour la prédiction du diabète")
plt.show()






































































