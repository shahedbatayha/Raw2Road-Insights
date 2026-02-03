import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


st.set_page_config(page_title="Used Cars DE Dashboard",layout="wide")
st.title("🚗 Raw2Road Insights dashboard")
st.markdown("---")


try:
    raw=pd.read_csv('raw_data.csv')
    clean=pd.read_csv('cleaned_data.csv')
except:
    st.error("run preprocessing.py first!")
    st.stop()


tab1,tab2,tab3,tab4,tab5=st.tabs(["📁 Raw data","📊 Visual analysis (EDA)","🔢 Target encoding","📏 Outliers & scaling","✅ Processed data"])

with tab1:
    st.header("1.Overview of raw dataset")
    c1,c2,c3=st.columns(3)
    c1.metric("Rows",raw.shape[0])
    c2.metric("Columns",raw.shape[1])
    c3.metric("Missing values",raw.isnull().sum().sum())
    st.dataframe(raw.head(15))

with tab2:
    st.header("2.Exploratory visualizations")
    col1,col2=st.columns(2)
    with col1:
        st.subheader("Correlation matrix")
        st.image('correlation_heatmap.png')
    with col2:
        st.subheader("Price distribution")
        st.image('price_distribution.png')
    col3,col4=st.columns(2)
    with col3:
        st.subheader("Top 10 Car brands")
        st.image('top_brands.png')
    with col4:
        st.subheader("Price vs Mileage Trend")
        st.image('price_vs_mileage.png')

with tab3:
    st.header("3.Target encoding")
    st.success("I replaced the categorical features by their mean price to avoid high dimensionality.")
    cx,cy=st.columns(2)
    with cx:
        st.subheader("Before (Categorical)")
        st.write(raw[['brand','fuel_type','transmission']].head(10))
    with cy:
        st.subheader("After (Target encoded)")
        encoded_cols=['brand_encoded','fuel_type_encoded','transmission_encoded']
        st.write(clean[encoded_cols].head(10))

with tab4:
    st.header("4.Outlier handling & scaling")
    st.info("Price and mileage is processed with IQR and StandardScaler.")
    c_a,c_b=st.columns(2)
    with c_a:
        st.subheader("Price before cleaning")
        raw_p=raw['price'].replace(r'[\$,]', '', regex=True).astype(float)
        fig1,ax1=plt.subplots();sns.boxplot(y=raw_p,color='#ff4b4b',ax=ax1);st.pyplot(fig1)
    with c_b:
        st.subheader("Price after scaling & outliers handling")
        fig2,ax2=plt.subplots();sns.boxplot(y=clean['price'],color='#00c853',ax=ax2);st.pyplot(fig2)

with tab5:
    st.header("5.Final processed and clean dataset")
    st.dataframe(clean.head(20))

st.sidebar.title("Project Info")
st.sidebar.info("Data Engineering course final project")
st.sidebar.write("by: shahed batayha")