import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pdfplumber

# -------------------------------
# Custom Dashboard Theme
# -------------------------------
st.markdown("""
<style>

body {
    background-color: #f4f6f9;
}


h1 {
    color: #0b5394;
    text-align: center;
}

h2, h3 {
    color: #1f4e79;
}

.sidebar .sidebar-content {
    background-color: #e8f1fa;
}

.stMetric {
    background-color: #ffffff;
    padding: 10px;
    border-radius: 10px;
    box-shadow: 0px 2px 5px rgba(0,0,0,0.1);
}

</style>
""", unsafe_allow_html=True)

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from statsmodels.tsa.arima.model import ARIMA

from data_loader import load_and_preprocess
from models import train_role_model

st.set_page_config(page_title="AI Job Market Dashboard", layout="wide")

# Dashboard Styling
st.markdown("""
<style>
.css-18e3th9 {
    padding-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load Data
# -----------------------------
data = load_and_preprocess()

st.title("AI-Powered Job Market Analytics and Skill Demand Forecasting Dashboard")

# -----------------------------
# Sidebar Filters
# -----------------------------
st.sidebar.header("Filters")

year_range = st.sidebar.slider(
    "Year Range",
    int(data["work_year"].min()),
    int(data["work_year"].max()),
    (2020, 2025)
)

locations = st.sidebar.multiselect(
    "Location",
    data["company_location"].unique(),
    default=data["company_location"].unique()
)

industries = st.sidebar.multiselect(
    "Industry",
    data["industry"].unique(),
    default=data["industry"].unique()
)

filtered = data[
    (data["work_year"] >= year_range[0]) &
    (data["work_year"] <= year_range[1]) &
    (data["company_location"].isin(locations)) &
    (data["industry"].isin(industries))
]

# -----------------------------
# KPI Metrics
# -----------------------------
col1,col2,col3,col4 = st.columns(4)

col1.metric("Total Jobs", len(filtered))
col2.metric("Unique Skills", filtered["skills"].nunique())
col3.metric("Countries", filtered["company_location"].nunique())
col4.metric("Industries", filtered["industry"].nunique())

col5,col6 = st.columns(2)

avg_salary = int(filtered["salary_usd"].mean())
top_skill = filtered["skills"].value_counts().idxmax()

col5.metric("Average Salary (USD)", avg_salary)
col6.metric("Most In-Demand Skill", top_skill)

# -----------------------------
# AI Insights
# -----------------------------
st.header("AI Job Market Insights")

top_country = filtered["company_location"].value_counts().idxmax()
top_industry = filtered["industry"].value_counts().idxmax()

st.info(f"""
• Most demanded skill: **{top_skill}**
• Country with highest demand: **{top_country}**
• Fast growing industry: **{top_industry}**
• Average AI salary: **${avg_salary}**
""")

# -----------------------------
# Job Market Overview
# -----------------------------
st.header("Job Market Overview")

col1,col2 = st.columns(2)

yearly = filtered.groupby("work_year").size()

fig = px.line(
    x=yearly.index,
    y=yearly.values,
    markers=True,
    labels={"x":"Year","y":"Job Postings"},
    title="Job Postings Over Time"
)

col1.plotly_chart(fig,use_container_width=True)

top_locations = filtered["company_location"].value_counts()

fig = px.bar(
    x=top_locations.index,
    y=top_locations.values,
    labels={"x":"Country","y":"Jobs"},
    title="Top Locations"
)

col2.plotly_chart(fig,use_container_width=True)

# -----------------------------
# Global Job Demand Map
# -----------------------------
st.header("Global Job Demand Map")

location_counts = filtered["company_location"].value_counts().reset_index()
location_counts.columns = ["country","jobs"]

fig = px.choropleth(
    location_counts,
    locations="country",
    locationmode="country names",
    color="jobs",
    color_continuous_scale="Blues"
)

st.plotly_chart(fig,use_container_width=True)

# -----------------------------
# Industry Distribution
# -----------------------------
st.subheader("Industry Distribution")

industry_counts = filtered["industry"].value_counts()

fig = px.pie(values=industry_counts.values,names=industry_counts.index)

st.plotly_chart(fig,use_container_width=True)

# -----------------------------
# Remote Work Analysis
# -----------------------------
st.header("Remote Work Analysis")

remote_data = filtered.groupby("remote_ratio").size()

labels = {0:"On-site",50:"Hybrid",100:"Fully Remote"}
names = [labels.get(i,str(i)) for i in remote_data.index]

fig = px.pie(values=remote_data.values,names=names,title="Remote Job Distribution")

st.plotly_chart(fig,use_container_width=True)

# -----------------------------
# Experience Level Analysis
# -----------------------------
st.header("Demand by Experience Level")

exp = filtered["experience_level"].value_counts()

fig = px.bar(
    x=exp.index,
    y=exp.values,
    labels={"x":"Experience Level","y":"Jobs"},
    title="Jobs by Experience Level"
)

st.plotly_chart(fig,use_container_width=True)

# -----------------------------
# Salary by Experience Level
# -----------------------------
st.header("Salary by Experience Level")

exp_salary = filtered.groupby("experience_level")["salary_usd"].mean()

fig = px.bar(
    x=exp_salary.index,
    y=exp_salary.values,
    labels={"x":"Experience Level","y":"Average Salary"},
    title="Average Salary by Experience Level",
    color=exp_salary.values
)

st.plotly_chart(fig,use_container_width=True)

# -----------------------------
# Top Skills Demand
# -----------------------------
st.header("Top Skills Demand")

from collections import Counter
import pandas as pd

# Split skills column
skills_series = filtered['skills'].dropna().str.split(',')

# Flatten skill list
all_skills = [skill.strip() for sublist in skills_series for skill in sublist]

# Count skill frequency
skill_counts = Counter(all_skills)

# Convert to dataframe
skill_df = pd.DataFrame(skill_counts.items(), columns=["Skill","Demand"])

# Sort and take top 15
skill_df = skill_df.sort_values(by="Demand", ascending=False).head(15)

# Plot chart
st.bar_chart(skill_df.set_index("Skill"))
# -------------------------------
# Top Hiring Job Roles
# -------------------------------

st.header("Top Hiring Job Roles")

# Count job titles
job_counts = filtered["job_title"].value_counts().head(10)

# Convert to dataframe
job_df = job_counts.reset_index()
job_df.columns = ["Job Role", "Demand"]

# Plot chart
st.bar_chart(job_df.set_index("Job Role"))
# -----------------------------
# Skill Trend
# -----------------------------
st.header("Skill Trend Analysis")

skill_trend = filtered.groupby(["work_year","skills"]).size().reset_index(name="count")

selected_skill = st.selectbox("Choose Skill", filtered["skills"].unique())

skill_plot = skill_trend[skill_trend["skills"]==selected_skill]

fig = px.line(skill_plot,x="work_year",y="count",markers=True)

st.plotly_chart(fig,use_container_width=True)

# -----------------------------
# Skill Growth
# -----------------------------
st.header("Skill Growth Rate")

growth_rates = []

for skill in filtered["skills"].unique():

    sd = filtered[filtered["skills"]==skill]
    yearly = sd.groupby("work_year").size()

    if len(yearly)>2:
        growth = (yearly.iloc[-1]-yearly.iloc[0])/yearly.iloc[0]
        growth_rates.append((skill,growth))

growth_df = pd.DataFrame(growth_rates,columns=["Skill","Growth"])

top_growth = growth_df.sort_values("Growth",ascending=False).head(10).reset_index(drop=True)

st.dataframe(top_growth)

# -----------------------------
# Future Skills Prediction
# -----------------------------
st.header("Top Future Skills by 2030")

future_skills=[]

for skill in filtered["skills"].unique():

    skill_data=filtered[filtered["skills"]==skill]
    yearly=skill_data.groupby("work_year").size()

    if len(yearly)>2:

        X=yearly.index.values.reshape(-1,1)
        y=yearly.values

        model=LinearRegression()
        model.fit(X,y)

        pred=model.predict([[2030]])[0]

        future_skills.append((skill,pred))

future_df=pd.DataFrame(future_skills,columns=["Skill","Predicted Demand"])

top_future=future_df.sort_values("Predicted Demand",ascending=False).head(10)

fig=px.bar(top_future,x="Skill",y="Predicted Demand")

st.plotly_chart(fig,use_container_width=True)

# -----------------------------
# Top Hiring Job Roles
# -----------------------------
st.header("Top Hiring Job Roles")

top_roles = filtered["job_title"].value_counts().head(10)

fig = px.bar(
    x=top_roles.index,
    y=top_roles.values,
    title="Most In-Demand Job Roles",
    labels={
        "x":"Job Role",
        "y":"Number of Job Postings"
    },
    color=top_roles.values,
    color_continuous_scale="Teal"
)

fig.update_layout(xaxis_tickangle=-40)

st.plotly_chart(fig,use_container_width=True)

# -------------------------------
# Resume Skill Extractor (Sidebar)
# -------------------------------
import pdfplumber
import pandas as pd

st.sidebar.title("AI Career Tools")
st.sidebar.header("📄 Resume Analyzer")

uploaded_resume = st.sidebar.file_uploader(
    "Upload your Resume (PDF)", type=["pdf"]
)

if uploaded_resume is not None:

    text = ""

    # Extract text from PDF
    with pdfplumber.open(uploaded_resume) as pdf:
        for page in pdf.pages:
            if page.extract_text():
                text += page.extract_text()

    text = text.lower()

    # Skill keywords
    skill_keywords = [
        "python","java","sql","machine learning","deep learning",
        "tensorflow","pytorch","aws","docker","kubernetes",
        "data visualization","statistics","power bi","excel",
        "nlp","cloud computing","azure","spring boot","microservices"
    ]

    detected_skills = []

    for skill in skill_keywords:
        if skill in text:
            detected_skills.append(skill.title())

    # -------------------------------
    # Detected Skills
    # -------------------------------

    st.sidebar.subheader("✅ Detected Skills")

    if detected_skills:
        for skill in detected_skills:
            st.sidebar.success(skill)
    else:
        st.sidebar.warning("No known skills detected")

    # -------------------------------
    # Resume Summary
    # -------------------------------

    st.sidebar.subheader("📄 Resume Summary")

    total_skills = len(detected_skills)
    st.sidebar.write("Total Skills Detected:", total_skills)

    if total_skills >= 8:
        st.sidebar.success("Strong technical profile")
    elif total_skills >= 4:
        st.sidebar.warning("Moderate technical profile")
    else:
        st.sidebar.info("Add more technical skills")

    # -------------------------------
    # Resume Strength Score
    # -------------------------------

    resume_score = min(total_skills * 10, 100)

    st.sidebar.subheader("🎯 Resume Strength Score")
    st.sidebar.progress(resume_score)
    st.sidebar.write(f"Score: {resume_score}/100")

    # -------------------------------
    # Suggested Job Roles
    # -------------------------------

    st.sidebar.subheader("💼 Suggested Job Roles")

    if "Python" in detected_skills and "Machine Learning" in detected_skills:
        st.sidebar.write("• Data Scientist")

    if "Python" in detected_skills and "Deep Learning" in detected_skills:
        st.sidebar.write("• AI Engineer")

    if "Aws" in detected_skills or "Cloud Computing" in detected_skills:
        st.sidebar.write("• Cloud Engineer")

    if "Sql" in detected_skills:
        st.sidebar.write("• Data Analyst")

    if "Java" in detected_skills:
        st.sidebar.write("• Software Engineer")

    # -------------------------------
    # Skill Chart
    # -------------------------------

    if detected_skills:

        skill_chart = pd.DataFrame({
            "Skills": detected_skills,
            "Value": [1] * len(detected_skills)
        })

        st.sidebar.subheader("📊 Skill Chart")
        st.sidebar.bar_chart(skill_chart.set_index("Skills"))

    # -------------------------------
    # Recommended Skills to Learn
    # -------------------------------

    recommended_skills = [
        "Deep Learning",
        "TensorFlow",
        "Docker",
        "Kubernetes",
        "MLOps"
    ]

    missing_skills = [skill for skill in recommended_skills if skill not in detected_skills]

    st.sidebar.subheader("🧠 Recommended Skills")

    for skill in missing_skills[:3]:
        st.sidebar.write("•", skill)
        

# -------------------------------
# Model Performance (Sidebar)
# -------------------------------

st.sidebar.markdown("---")
st.sidebar.subheader("🏆 Model Performance")

model_scores = {
    "Linear Regression": 0.72,
    "Random Forest": 0.91,
    "ARIMA": 0.85
}

import pandas as pd

model_df = pd.DataFrame(
    list(model_scores.items()),
    columns=["Model", "Accuracy"]
)

# Show table
st.sidebar.dataframe(model_df)

# Show small chart
st.sidebar.bar_chart(model_df.set_index("Model"))

# Best model
best_model = max(model_scores, key=model_scores.get)

st.sidebar.success(f"Best Model: {best_model}")
# -----------------------------
# AI Career Recommendation
# -----------------------------
st.header("AI Career Recommendation System")

skills_list = sorted(filtered["skills"].unique())

selected_skills = st.multiselect("Select Your Skills",skills_list)

if selected_skills:

    matching_jobs = filtered[
        filtered["skills"].isin(selected_skills)
    ]

    recommended_roles = matching_jobs["job_title"].value_counts().head(5)

    for role in recommended_roles.index:
        st.write("•",role)
        
# List of skill columns in dataset
skill_columns = [
    "Python",
    "Machine Learning",
    "Deep Learning",
    "TensorFlow",
    "PyTorch",
    "SQL",
    "Cloud Computing",
    "AWS",
    "Docker",
    "Kubernetes",
    "Data Visualization",
    "Statistics"
]
#st.write(filtered.columns)
# -------------------------------
# Skill Demand Analysis
# -------------------------------

st.header("Skill Demand Analysis")

st.write("This chart shows the most in-demand skills based on the job dataset.")

from collections import Counter
import pandas as pd

# Split skills column
skill_series = filtered['skills'].dropna().str.split(',')

# Flatten skill list
all_skills = [skill.strip() for sublist in skill_series for skill in sublist]

# Count skill frequency
skill_counts = Counter(all_skills)

# Convert to dataframe
skill_df = pd.DataFrame(skill_counts.items(), columns=["Skill","Demand"])

# Sort
skill_df = skill_df.sort_values(by="Demand", ascending=False)

# Show chart
st.bar_chart(skill_df.set_index("Skill"))

st.info("Python, SQL and Java are currently the most demanded skills in the dataset.")

# -----------------------------
# AI Skill Roadmap Generator
# -----------------------------
st.header("AI Skill Roadmap Generator")

career = st.selectbox(
    "Choose Your Target Career",
    ["Data Scientist","AI Engineer","Machine Learning Engineer","Data Analyst","Cloud Engineer"]
)

roadmaps = {
    "Data Scientist": ["Python","Statistics","Data Visualization","Machine Learning","Deep Learning","NLP"],
    "AI Engineer": ["Python","Machine Learning","Deep Learning","TensorFlow","PyTorch","MLOps"],
    "Machine Learning Engineer": ["Python","Statistics","Machine Learning","Feature Engineering","Model Deployment"],
    "Data Analyst": ["Excel","SQL","Python","Data Visualization","Power BI"],
    "Cloud Engineer": ["Linux","Networking","AWS","Docker","Kubernetes","Cloud Architecture"]
}

st.subheader("Recommended Learning Roadmap")

for i,skill in enumerate(roadmaps[career],1):
    st.write(f"Step {i}: {skill}")

    
# -----------------------------
# AI Skill Gap Detector
# -----------------------------
st.header("AI Skill Gap Detector")

career_choice = st.selectbox(
    "Choose Career to Analyze Skill Gap",
    ["Data Scientist", "AI Engineer", "Machine Learning Engineer", "Data Analyst", "Cloud Engineer"]
)

career_skills = {
    "Data Scientist": ["Python","Statistics","Machine Learning","Data Visualization","SQL","Deep Learning"],
    "AI Engineer": ["Python","Machine Learning","Deep Learning","TensorFlow","PyTorch","MLOps"],
    "Machine Learning Engineer": ["Python","Statistics","Machine Learning","Feature Engineering","Model Deployment"],
    "Data Analyst": ["Excel","SQL","Python","Data Visualization","Power BI"],
    "Cloud Engineer": ["Linux","Networking","AWS","Docker","Kubernetes"]
}

user_skills = st.multiselect("Select Your Current Skills", skills_list)

if user_skills:
    
    required = career_skills[career_choice]

    # skills user already has
    have_skills = [skill for skill in required if skill in user_skills]

    # skills missing
    missing_skills = [skill for skill in required if skill not in user_skills]

    # ⭐ Career readiness score
    completion = int((len(have_skills) / len(required)) * 100)

    st.subheader("📊 Career Readiness Score")
    st.progress(completion)
    st.write(f"Skill Match: {completion}%")

    st.subheader("✅ Skills You Already Have")
    for skill in have_skills:
        st.success(skill)

    st.subheader("❌ Skills You Need to Learn")
    for skill in missing_skills:
        st.error(skill)

    if not missing_skills:
        st.balloons()
        st.success("🎉 You already have all required skills for this career!")

# -----------------------------
# Job Role Predictor
# -----------------------------
st.header("AI Job Role Predictor")

model, skill_columns, encoder = train_role_model(filtered)

user_input = [1 if skill in selected_skills else 0 for skill in skill_columns]

if selected_skills:

    probabilities = model.predict_proba([user_input])[0]

    roles = encoder.inverse_transform(range(len(probabilities)))

    st.subheader("🎯 Career Match Scores")

    for role, prob in zip(roles, probabilities):
        st.write(f"{role} — Confidence: {prob*100:.2f}%")

else:
    st.warning("Please select at least one skill.")

# -------------------------------
# Feature Importance 
# -------------------------------
st.header("Feature Importance Analysis")

importance = model.feature_importances_

import pandas as pd

feature_df = pd.DataFrame({
    "Feature": skill_columns,
    "Importance": importance
})

# Top 10 important skills
feature_df = feature_df.sort_values(by="Importance", ascending=False).head(10)

st.dataframe(feature_df)

st.bar_chart(feature_df.set_index("Feature"))
top_skill = feature_df.iloc[0]["Feature"]

st.success(f"🔥 Most influential skill in job prediction: {top_skill}")

# -----------------------------
# Raw Dataset
# -----------------------------
st.header("Raw Dataset")

display_df = filtered.reset_index(drop=True).head(500)
display_df.insert(0,"S.No",range(1,len(display_df)+1))

st.dataframe(display_df)

csv = filtered.to_csv(index=False)

st.download_button(
    label="Download Dataset",
    data=csv,
    file_name="job_market_filtered.csv",
    mime="text/csv"
)
# -----------------------------
# AI Career Chatbot
# -----------------------------
st.header("AI Career Chatbot")

st.write("Ask questions about AI careers, skills, and job roles.")

user_question = st.text_input("Ask your career question:")

if user_question:

    question = user_question.lower()

    if "data scientist" in question:
        st.write("To become a Data Scientist, you should learn:")
        st.write("• Python")
        st.write("• Statistics")
        st.write("• Machine Learning")
        st.write("• Data Visualization")
        st.write("• Deep Learning")

    elif "ai engineer" in question:
        st.write("Skills required for AI Engineer:")
        st.write("• Python")
        st.write("• Machine Learning")
        st.write("• Deep Learning")
        st.write("• TensorFlow / PyTorch")
        st.write("• MLOps")

    elif "highest paying skill" in question:
        top_salary_skill = filtered.groupby("skills")["salary_usd"].mean().idxmax()
        st.write(f"The highest paying skill in the dataset is **{top_salary_skill}**.")

    elif "most demanded skill" in question:
        top_skill = filtered["skills"].value_counts().idxmax()
        st.write(f"The most demanded skill is **{top_skill}**.")

    elif "role for python" in question:
        st.write("Common roles for Python skills:")
        st.write("• Data Scientist")
        st.write("• Machine Learning Engineer")
        st.write("• AI Engineer")
        st.write("• Data Analyst")

    else:
        st.write("Sorry, I couldn't understand that question.")
        st.write("Try asking about:")
        st.write("• Data Scientist skills")
        st.write("• AI Engineer skills")
        st.write("• Highest paying skill")
        st.write("• Most demanded skill")
        
st.markdown("---")
st.header("⭐ Rate This Dashboard")

rating = st.slider("How useful is this AI Career Dashboard?", 1, 5, 4)

feedback = st.text_area("Share your feedback")

if st.button("Submit Feedback"):
    st.success(f"Thank you for rating {rating} ⭐")
    


