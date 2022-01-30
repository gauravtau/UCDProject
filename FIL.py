#import re
#print(re.sub(r"!",",","WOW!what a nice day"))

# IMPORTING DATA: CALLING API
import requests
request=requests.get("http://api.open-notify.org/iss-now.json")
dict=request.json()
print(dict)
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
print(a["ID"])    #marking as comments to not invoke the print command

#import pandas as pd
#import matplotlib.pyplot as plt
#import seaborn as sns
#fig,ax = plt.subplots()
#data= pd.read_csv("GBvideos.csv")
#x = data['channel_title'].head(5)
#y1 = data['views'].head(5)
#y2 = data['likes'].head(5)
#ax.bar(x,y1) #or plt.plot(x,y1)
#ax.plot(x,y2, marker="v") #or plt.plot(x,y2)
#sns.barplot(x,y1)
#sns.lineplot(x,y2)
#print(x,y1,y2)
#plt.show()