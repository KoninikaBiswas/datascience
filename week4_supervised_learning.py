import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

df = pd.read_csv("online_retail_cleaned.csv")
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
sales = df[(~df["IsCancellation"]) & (df["CustomerID"].notna()) & (df["Quantity"]>0) & (df["UnitPrice"]>0)].copy()
sales["SalesAmount"] = sales["Quantity"] * sales["UnitPrice"]
analysis_date = sales["InvoiceDate"].max() + pd.Timedelta(days=1)
customer = sales.groupby("CustomerID").agg(
 Recency=("InvoiceDate", lambda x:(analysis_date-x.max()).days),
 Frequency=("InvoiceNo","nunique"),
 Monetary=("SalesAmount","sum"),
 AverageOrderValue=("SalesAmount","mean"),
 AverageQuantity=("Quantity","mean"),
 AverageUnitPrice=("UnitPrice","mean"),
 Country=("Country","first")
).reset_index()
threshold = customer["Monetary"].quantile(.75)
customer["HighValue"] = (customer["Monetary"] >= threshold).astype(int)
features=["Recency","Frequency","AverageOrderValue","AverageQuantity","AverageUnitPrice","Country"]
X,y=customer[features],customer["HighValue"]
pre=ColumnTransformer([("num","passthrough",features[:-1]),("cat",OneHotEncoder(handle_unknown="ignore"),["Country"])])
model=Pipeline([("preprocessor",pre),("classifier",RandomForestClassifier(n_estimators=300,min_samples_leaf=2,class_weight="balanced",random_state=42,n_jobs=-1))])
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.20,random_state=42,stratify=y)
model.fit(X_train,y_train)
pred=model.predict(X_test); prob=model.predict_proba(X_test)[:,1]
print("Accuracy:",accuracy_score(y_test,pred))
print("Precision:",precision_score(y_test,pred,zero_division=0))
print("Recall:",recall_score(y_test,pred,zero_division=0))
print("F1:",f1_score(y_test,pred,zero_division=0))
print("ROC-AUC:",roc_auc_score(y_test,prob))
cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
print(cross_validate(model,X_train,y_train,cv=cv,scoring=["accuracy","precision","recall","f1","roc_auc"]))