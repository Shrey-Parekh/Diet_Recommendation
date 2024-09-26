import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def calculate_bmi(height, weight):
    if height == 0:
        return 0
    return round(weight / ((height / 100) ** 2), 2)

def get_body_type(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif 18.5 <= bmi < 24.9:
        return "Normal Weight"
    elif 25 <= bmi < 29.9:
        return "Overweight"
    else:
        return "Obese"

def recommend_food(height, weight, diet_preference, veg_preference):
    bmi = calculate_bmi(height, weight)
    body_type = get_body_type(bmi)

    user_input = scaler.transform([[height, weight, 0, 0]])
    rf_recommendation = rf_classifier.predict(user_input)[0]
    svm_recommendation = svm_classifier.predict(user_input)[0]

    filtered_df = df.copy()
    if veg_preference == 1:
        filtered_df = filtered_df[filtered_df['Vegetarian'] == 1]
    else:
        filtered_df = filtered_df[filtered_df['Vegetarian'] == 0]

    recommended_foods = []

    if body_type == "Normal Weight":
        if diet_preference == "Weight Loss":
            recommended_foods = filtered_df[(filtered_df['Type'] == rf_recommendation) &
                                            (filtered_df['Carbs(g)'] < 15)]["Food"].tolist()[:7]
        elif diet_preference == "Weight Gain":
            recommended_foods = filtered_df[(filtered_df['Type'] == svm_recommendation) &
                                            (filtered_df['Carbs(g)'] >= 20) &
                                            (filtered_df['Protein(g)'] >= 15)]["Food"].tolist()[:7]
        else:
            recommended_foods = filtered_df[(filtered_df['Protein(g)'] >= 10) &
                                            (filtered_df['Carbs(g)'] < 25) &
                                            (filtered_df['Fat(g)'] <= 20) &
                                            (filtered_df['Calories'] < 300)]["Food"].tolist()[:7]
    elif body_type == "Overweight":
        recommended_foods = filtered_df[(filtered_df['Type'] == rf_recommendation) &
                                        (filtered_df['Carbs(g)'] < 20)]["Food"].tolist()[:7]
    elif body_type == "Underweight":
        recommended_foods = filtered_df[(filtered_df['Type'] == svm_recommendation) &
                                        (filtered_df['Protein(g)'] >= 12)]["Food"].tolist()[:7]

    if not recommended_foods:
        recommended_foods = filtered_df["Food"].tolist()[:7]

    return recommended_foods, bmi, body_type

def plot_nutritional_trends(recommended_foods):
    if recommended_foods:
        nutrient_data = df[df['Food'].isin(recommended_foods)][['Food', 'Calories', 'Protein(g)', 'Carbs(g)', 'Fat(g)']]
        nutrient_data.set_index('Food', inplace=True)
        nutrient_data.plot(kind='bar', figsize=(10, 5))
        plt.title("Nutritional Content of Recommended Foods")
        plt.xlabel("Food")
        plt.ylabel("Nutritional Value")
        plt.xticks(rotation=45)
        st.pyplot(plt)

df = pd.read_csv("food.csv")
df = df.drop_duplicates()
label_encoder = LabelEncoder()
df['Type'] = label_encoder.fit_transform(df['Type'])

scaler = StandardScaler()
X = scaler.fit_transform(df[['Calories', 'Protein(g)', 'Carbs(g)', 'Fat(g)']])
y = df['Type']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_classifier = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_classifier.fit(X_train, y_train)

svm_classifier = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm_classifier.fit(X_train, y_train)

st.title("Food Recommendation System")

height = st.number_input("Enter Height (cm):", min_value=0.0)
weight = st.number_input("Enter Weight (kg):", min_value=0.0)
diet_preference = st.selectbox("Select Diet Preference:", ["Weight Loss", "Weight Gain", "Healthy Diet"])
veg_preference = st.radio("Select Vegetarian Preference:", (1, 0))

if st.button("Recommend Food"):
    recommended_foods, bmi, body_type = recommend_food(height, weight, diet_preference, veg_preference)

    st.write(f"BMI: {bmi}")
    st.write(f"Body Type: {body_type}")

    if recommended_foods:
        st.write("Recommended Food:")
        for food in recommended_foods:
            st.write(food)
        plot_nutritional_trends(recommended_foods)
    else:
        st.write("No recommendations available based on your inputs.")
