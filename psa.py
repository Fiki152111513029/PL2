import numpy as np
import pandas as pd
from statistics import mean
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.datasets import load_iris
import seaborn as sns

data = pd.read_csv("./Nutrions.csv")

# ambil atribut dalam variabel dataMissing
dataMissing = data.loc[:, ["Sodium_Content", "GI", "Veg_noVeg", "Calories", "Fats", "Protein", "Iron", "Calcium", "Sodium", "Potassium", "Carbohydrates", "Fiber", "Vitamin_D", "Sugars"]]
print(dataMissing.head())
print()

# mendeteksi missing value
print("Deteksi Missing Value")
print(dataMissing.isna().sum())
print("\n")

print("Hapus Missing Value")
dataMissing = dataMissing.dropna()
print(dataMissing.isna().sum())
print("\n")

print("Deteksi Outlier")
outliers = []
def detect_outlier(data):
    threshold = 3
    mean = np.mean(data)
    std = np.std(data)
    
    for x in data:
        z_score = (x-mean)/std
        if np.abs(z_score)>threshold:
            outliers.append(x)
    return outliers

# Mencetak Outlier
outlier1 = detect_outlier(data['Sodium_Content'])
print("outlier kolom Sodium_Content: ",outlier1)
print("banyak outlier Sodium_Content: ",len(outlier1)) 
print()

outlier2 = detect_outlier(data['GI']) 
print("outlier kolom GI': ",outlier2)
print("banyak outlier GI': ",len(outlier2)) 
print()

outlier3 = detect_outlier(data['Veg_noVeg']) 
print("outlier kolom Veg_noVeg': ",outlier3)
print("banyak outlier Veg_noVeg': ",len(outlier3)) 
print()

outlier4 = detect_outlier(data['Calories']) 
print("outlier kolom Calories': ",outlier4)
print("banyak outlier Calories': ",len(outlier4)) 
print()

outlier5 = detect_outlier(data['Fats']) 
print("outlier kolom Fats': ",outlier5)
print("banyak outlier Fats': ",len(outlier5)) 
print()

outlier6 = detect_outlier(data['Protein']) 
print("outlier kolom Protein': ",outlier6)
print("banyak outlier Protein': ",len(outlier6)) 
print()

outlier7 = detect_outlier(data['Iron']) 
print("outlier kolom Iron': ",outlier7)
print("banyak outlier Iron': ",len(outlier7)) 
print()

outlier8 = detect_outlier(data['Calcium']) 
print("outlier kolom Calcium': ",outlier8)
print("banyak outlier Calcium': ",len(outlier8)) 
print()

outlier9 = detect_outlier(data['Sodium']) 
print("outlier kolom Sodium': ",outlier9)
print("banyak outlier Sodium': ",len(outlier9)) 
print()

outlier10 = detect_outlier(data['Potassium']) 
print("outlier kolom Potassium': ",outlier10)
print("banyak outlier Potassium': ",len(outlier10)) 
print()

outlier11 = detect_outlier(data['Carbohydrates']) 
print("outlier kolom Carbohydrates': ",outlier11)
print("banyak outlier Carbohydrates': ",len(outlier11)) 
print()

outlier12 = detect_outlier(data['Fiber']) 
print("outlier kolom Fiber': ",outlier12)
print("banyak outlier Fiber': ",len(outlier12)) 
print()

outlier13 = detect_outlier(data['Vitamin_D']) 
print("outlier kolom Vitamin_D': ",outlier13)
print("banyak outlier Vitamin_D': ",len(outlier13)) 
print()

outlier14 = detect_outlier(data['Sugars']) 
print("outlier kolom Sugars': ",outlier14)
print("banyak outlier Sugars': ",len(outlier14)) 
print()



# Penanganan Outlier
variabel = ["Sodium_Content", "GI", "Veg_noVeg", "Calories", "Fats", "Protein", "Iron", "Calcium", "Sodium", "Potassium", "Carbohydrates", "Fiber", "Vitamin_D", "Sugars"] 
for var in variabel:
    outlier_datapoints = detect_outlier(data[var])
    print("Outlier ", var, " = ", outlier_datapoints) 
    rata = mean(data[var])
    print("Outlier ", var, "telah diganti menjadi mean : ") 
    data[var] = data[var].replace(outlier_datapoints, rata) 
    print(data)

print("===================================================================")
print("\n")

# # z-score
# zscore = stats.zscore(data, axis= 1)
# print("Hasil z-score =")
# print(zscore)
# print("============================================================")


#grouping yang diSugars_clustering menjadi dua
print("GROUPING VARIABEL".center(75,"="))
X=data.iloc[:,0:5].values
y=data.iloc[:,5].values
print("data variabel".center(75,"="))
print(X)
print("data kelas".center(75,"="))
print(y)
print("============================================================")
print("\n")

#Sugars_clusteringan training dan testing
print("SPLITTING DATA 20-80".center(75,"="))
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
print("instance variabel data training".center(75,"="))
print(X_train)
print("\n")
print("instance kelas data training".center(75,"="))
print(y_train)
print("\n")
print("instance variabel data testing".center(75,"="))
print(X_test)
print("\n")
print("instance kelas data testing".center(75,"="))
print(y_test)
print("============================================================")
print("\n")



# Misalkan kolom fitur Anda adalah 'Calories', 'Protein', 'Fat', 'Carbohydrates'
feature_columns = ['Sodium_Content', 'GI']

# Memisahkan fitur
X = data[feature_columns]

# Standarisasi fitur
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Melakukan hierarchical clustering
linked = linkage(X_scaled, method='ward')
# Menentukan cluster dengan fcluster
num_clusters = 2
clusters = fcluster(linked, num_clusters, criterion='maxclust')
data['Cluster'] = clusters

sns.clustermap(X_scaled, method='ward', col_cluster=False, cmap="viridis")
plt.show()

# Melihat distribusi data pada setiap cluster
print(data['Cluster'].value_counts())

# Menganalisis statistik deskriptif untuk setiap cluster
for cluster_num in range(1, num_clusters + 1):
    print(f"\nCluster {cluster_num} statistics:")
    print(data[data['Cluster'] == cluster_num][feature_columns].describe())

# Visualisasi
sns.pairplot(data, hue='Cluster', vars=feature_columns)
plt.show()

# Melihat distribusi data pada setiap cluster
print(data['Cluster'].value_counts())

# Menganalisis statistik deskriptif untuk setiap cluster
for cluster_num in range(1, num_clusters + 1):
    print(f"\nCluster {cluster_num} statistics:")
    print(data[data['Cluster'] == cluster_num][feature_columns].describe())

# Visualisasi
sns.pairplot(data, hue='Cluster', vars=feature_columns)
plt.show()

print(data[['Food_item', 'Cluster']])

feature_columns = ['Sodium','Calories', 'Fats', 'Protein', 'Iron', 'Calcium', 'Potassium', 'Carbohydrates', 'Fiber', 'Vitamin_D']
target_colomns = 'Cluster'

X = data[feature_columns]
y = data[target_colomns]


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Initialize the Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=0)
# Train the model
rf.fit(X_train, y_train)

# Make predictions
y_pred = rf.predict(X_test)



# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')
conf_matrix = confusion_matrix(y_test, y_pred)

print('CLASSIFICATION REPORT RANDOM FOREST'.center(75,'='))
print(classification_report(y_test, y_pred))

# Print evaluation metrics
print("Accuracy:", accuracy * 100, "%" )
print("Precision:" + str (precision))
print("Recall:" + str (recall))
print("F1 Score:" + str (f1))
print("Confusion Matrix:\n", conf_matrix)

custom_labels = ['1', '2']

# Membuat heatmap
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",xticklabels=custom_labels, yticklabels=custom_labels)

# Plot confusion matrix
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# REKOMENDASI MENGGUNAKAN RANDOMFOREST
feature_columns = ['Cluster']
target_colomns = 'Food_item'

X = data[feature_columns]
y = data[target_colomns]


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Initialize the Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=0)
# Train the model
rf.fit(X_train, y_train)

# Make predictions
y_pred = rf.predict(X_test)

def rekomendasi_makanan(Food_list):
    max_sodium = 1  # Set the calcium threshold inside the function
    filtered_foods = data[data['Sodium_Content'] <= max_sodium]['Food_item'].tolist()
    return [food for food in Food_list if food in filtered_foods]

print(("rekomendasi makanan untuk penderita hipertensi"))
age = int(input("Age = "))
vegnoveg = int(input("Veg/NoVeg (1/0)= "))
systolic = int(input("Tekanan Systolic (in mmHg) = "))
diastolic = int(input("Tekanan Diastolik (in mmHg) = "))

# fungsi rekomendasi_makanan

if age < 60:
    if systolic <= 120 and diastolic < 80 :
            print("bebek")
            Cluster = 1
    elif systolic <= 129 and diastolic <= 84 :
            print("Dianjurkan untuk memilih makanan rendah garam dan rendah lemak.")
            Cluster = 1
    elif systolic <= 139 and diastolic <= 89 :
            print("Dianjurkan untuk memilih makanan rendah garam dan rendah lemak.")
            Cluster = 1
    elif systolic <= 159 and diastolic <= 99 :
            print("Dianjurkan untuk memilih makanan rendah garam dan rendah lemak.")
            Cluster = 1  
    elif systolic <= 180 and diastolic <= 110 :
            print("Dianjurkan untuk memilih makanan rendah garam dan rendah lemak.")
            Cluster = 1 
new_user = pd.DataFrame({
     'Cluster': [Cluster],
})

# Predict the recommended food
recommended_foods = rf.predict(new_user)

filtered_recommendations = rekomendasi_makanan (recommended_foods)

if filtered_recommendations:
    print('Recommended Foods:')
    for food in filtered_recommendations:
        print(f'- {food}')
else:
    print('No suitable low-calcium recommendation found.')