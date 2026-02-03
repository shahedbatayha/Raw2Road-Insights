import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

df=pd.read_csv(r"C:\Users\Pc\sdk DE course\Used Car Price.csv")
raw_df=df.copy()


print("Exploratory Data Analysis (EDA)")
print(f"Data Shape:{df.shape}")
df['price']=df['price'].replace(r'[\$,]', '', regex=True).astype(float)
df['milage']=df['milage'].replace(r'[ mi.,]', '', regex=True).astype(float)
print("\n Descriptive Statistics \n",df.describe().T)

plt.figure(figsize=(10,6))
sns.heatmap(df.select_dtypes(include=[np.number]).corr(),annot=True,cmap='coolwarm',fmt='.2f')
plt.title('Correlation Heatmap')
plt.savefig('correlation_heatmap.png')
plt.figure(figsize=(10,5))
sns.histplot(df['price'],kde=True,color='blue')
plt.title('Price Distribution')
plt.savefig('price_distribution.png')
plt.figure(figsize=(10,5))
sns.scatterplot(data=df,x='milage',y='price',alpha=0.5)
plt.title('Price vs Mileage')
plt.savefig('price_vs_mileage.png')
plt.figure(figsize=(12,6))
df['brand'].value_counts().head(10).plot(kind='bar',color='skyblue')
plt.title('Top 10 Brands')
plt.xticks(rotation=45)
plt.savefig('top_brands.png')


df['fuel_type']=df['fuel_type'].fillna('Unknown')
df['accident']=df['accident'].fillna('None reported')
df['clean_title']=df['clean_title'].fillna('No')
df.dropna(inplace=True)

for col in ['price','milage']:
    Q1=df[col].quantile(0.25)
    Q3=df[col].quantile(0.75)
    IQR=Q3-Q1
    df[col]=np.where(df[col]>Q3+1.5*IQR,Q3+1.5*IQR,np.where(df[col]<Q1-1.5*IQR,Q1-1.5*IQR,df[col]))

target_cols=['brand','model','fuel_type','transmission']
for col in target_cols:
    target_mean=df.groupby(col)['price'].mean()
    df[f'{col}_encoded']=df[col].map(target_mean)


scaler=StandardScaler()
num_cols=['milage','price']
df[num_cols]=scaler.fit_transform(df[num_cols])
raw_df.to_csv('raw_data.csv',index=False)
df.to_csv('cleaned_data.csv',index=False)
