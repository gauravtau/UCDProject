
# IMPORTING DATA: CALLING API
import requests
request=requests.get("http://api.open-notify.org/iss-now.json")
dict=request.json()
# print(dict)     #commenting so that the print command does not get executed
request1=requests.get("http://api.open-notify.org/astros.json")
data=request1.json()
 #for p in data['people']:print(p['name'])    #commenting so that the print command does not get executed
# END

# IMPORTING DATA: IMPORTING CSV
import pandas as pd
data=pd.read_csv("netflix_titles.csv")
# END

# ANALYSING DATA: using Regex to extract a pattern in data
import re
p_number=(re.findall("\d\d\d-\d\d\d-\d\d\d\d","My phone number is 415-555-2352"))
#print(p_number)   #marking as comments to not invoke the print command
# END

# ANALYSING DATA: replace missing values and drop duplicates
data1=data.fillna(method="bfill").fillna(method="ffill")  # Replace missing values
data2=data.drop_duplicates() #drop duplicates
# END

# ANALYSING DATA: Iterators
data3 = ("Gaurav", "Shubham", "Ankit")
for x in data3:
 #print(x)    #marking as comments to not invoke the print command
# END

# ANALYSING DATA: Merge Dataframes
    canadian=pd.read_csv("CAvideos.csv")
    british=pd.read_csv("GBvideos.csv")
concat=pd.concat([canadian,british])    #concatenate
left=canadian.set_index(['title','trending_date'])
right=british.set_index(['title','trending_date'])
join=left.join(right,lsuffix="_CA", rsuffix="_GB")   #join
merge=pd.merge(left,right,on="video_id")    #merge
#print(left.shape,right.shape,merge.shape)    #marking as comments to not invoke the print command
# END

# PYTHON: Define custom function
def add(a, b):
    return a + b
def is_true(a):
    return bool(a)
res = add(2, 3)
#print("Result of add function is", res)   #marking as comments to not invoke the print command
res = is_true(2 < 5)
#print("Result of is_true function is",res)    #marking as comments to not invoke the print command
# END

# PYTHON: DICTIONARY
a={"ID": [1,2],"Name": ["Gaurav","Shubham"],"Marks": [85,95]}
# print(a["Name"])    #marking as comments to not invoke the print command
# END

# VISUALISE
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")
data_plot = pd.read_csv("GBvideos.csv")
x = data_plot['video_id'].head(5)
y1 = data_plot['views'].head(5)
y2 = data_plot['likes'].head(5)
sns.lineplot(x, y1, linestyle="--")    #line chart
fig,ax = plt.subplots()
sns.barplot(x, y2)    #bar chart
data_plot1 = pd.read_csv("processed.cleveland2.csv")
x2 = data_plot1['num'].head(5)
y3 = data_plot1['age'].head(5)
fig,ax = plt.subplots()
sns.scatterplot(x2, y3, marker="+", legend="full")    #scatter chart

# MACHINE LEARNING: Supervised
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np

#predict target variables
patient_data =pd.read_csv("processed.cleveland2.csv")
y=pd.DataFrame(patient_data["num"].values).to_numpy()
X=pd.DataFrame(patient_data.drop("num",axis=1).values).to_numpy()
knn=KNeighborsClassifier(n_neighbors=17)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=20, stratify=y)
knn.fit(X_train,y_train.ravel())
#print(knn.score(X_train,y_train))
predict=knn.predict(X_test)
#print(knn.score(X_test,y_test))
#end

#find the best count of neighbors to be used using chart
neighbors = np.arange(1, 20)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))
for i, k in enumerate(neighbors):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train)
    train_accuracy[i] = knn.score(X_train,y_train)
    test_accuracy[i] = knn.score(X_test,y_test)
plt.figure()
plt.title('k-NN: Varying Number of Neighbors')
plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()
#end
# END
