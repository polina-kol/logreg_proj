import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

st.title("📉 Сервис анализа данных с логистической регрессией")


uploaded_file = st.file_uploader("Загрузите CSV файл", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("Файл успешно загружен!")


    target_col = st.selectbox("Выберите целевую переменную (binary)", df.columns)

    if df[target_col].nunique() != 2:
        st.warning("Целевая переменная должна быть бинарной (0 и 1)")
    else:

        feature_cols = st.multiselect("Выберите фичи для обучения", df.columns.drop(target_col))

        if len(feature_cols) < 1:
            st.warning("Выберите хотя бы одну фичу для обучения.")
        else:
            X = df[feature_cols]
            y = df[target_col]


            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)


            model = LogisticRegression()
            model.fit(X_scaled, y)


            st.subheader("🔍 Коэффициенты модели:")
            coef_dict = {col: coef for col, coef in zip(feature_cols, model.coef_[0])}
            coef_dict['intercept'] = model.intercept_[0]
            st.json(coef_dict)


            st.subheader("📊 Построить scatter plot")

            col1, col2 = st.columns(2)
            with col1:
                x_axis = st.selectbox("Ось X", feature_cols)
            with col2:
                y_axis = st.selectbox("Ось Y", feature_cols)

            if x_axis == y_axis:
                st.warning("Выберите разные оси для построения графика.")
            else:
                plt.figure(figsize=(8, 6))
                classes = df[target_col].unique()

                for cls in classes:
                    subset = df[df[target_col] == cls]
                    plt.scatter(subset[x_axis], subset[y_axis],
                                label=f'Class {cls}', alpha=0.7)

                plt.xlabel(x_axis)
                plt.ylabel(y_axis)
                plt.title(f"{x_axis} vs {y_axis}, цвет — значение таргета")
                plt.legend()
                plt.grid(True)
                st.pyplot(plt)

else:
    st.info("Пожалуйста, загрузите CSV файл.")