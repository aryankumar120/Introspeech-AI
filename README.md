# 🟣 IntroSpeech AI
AI-Powered Student Introduction Evaluation System

IntroSpeech AI is an interactive Streamlit application that evaluates a student’s self-introduction using a structured rubric.
It combines semantic similarity, keyword matching, and structured scoring to generate detailed, weighted feedback.

The system aims to help teachers, schools, and training institutes objectively evaluate student introductions using AI.

## 🌟 Key Features
🔹 Uploadable Rubric

Upload any .xlsx rubric file.
Supports:

Criterion

Description

Keywords

Weight

A clean default rubric is provided.

## 🔹 Transcript Analyzer

Paste a student’s self-introduction and receive:

Weighted overall score

Semantic similarity score

Keyword match score

Structural score

## 🔹 Detailed Breakdown

For each criterion:

Meaning

Extracted keywords

Matched keywords (highlighted)

Semantic score

Structure score

Total weighted score

Clean UI cards styled in a lavender theme

## 🔹 Downloadable Results

Download a CSV that contains:

All criteria

Raw scores

Similarity

Structure score

Weight

Keywords matched

# 🧠 How It Works
1. Rubric Processing

The application reads the uploaded rubric using Pandas.

Each row contains:

criterion

description

keywords (comma-separated)

weight

2. Transcript Parsing

The user pastes a self-introduction into the text area.

The system:

Cleans text

Converts to lowercase

Splits into tokens for keyword checks

3. Semantic Scoring

Uses:

SentenceTransformer("all-MiniLM-L6-v2")


This computes cosine similarity between:

Criterion keywords

Student transcript

Scores range between 0 and 1, later scaled to weights.

4. Keyword Matching

Checks if transcript contains any defined keywords.

5. Structure Scoring

Ensures flow:
Greeting → Personal → Family → Hobbies → Fun Fact → Favorite Subject → Closing

Gives a boost if student maintains expected order.

6. Weighted Score Calculation
final_score = (semantic + keyword_score + structure_score) * weight

7. Visualization

Using Plotly:

Horizontal bar chart for criterion scores

Pie chart for score distribution


## 📂 Project Structure
├── app.py
├── rubric_clean.xlsx
├── requirements.txt
├── README.md (this file)
└── (generated) results.csv

## 🚀 Installation
1. Create a virtual environment
python3 -m venv venv
source venv/bin/activate

2. Install dependencies
pip install -r requirements.txt

3. Run the application
streamlit run app.py

📦 requirements.txt

Use this for your submission:

streamlit
pandas
numpy
sentence-transformers
torch
scikit-learn
plotly
openpyxl

## 📘 Example Usage

Open Streamlit

Upload rubric_clean.xlsx

Paste transcript:

Hello everyone, my name is...


Click Analyze Speech

View:

KPIs

Graphs

Criterion-level breakdown

Download CSV

## 🏆 Why This Project Stands Out
✔ Complete end-to-end pipeline
✔ Real AI evaluation using embeddings
✔ Weighted scoring mimics real academics
✔ Cleanest UI possible
✔ Recruiter and assignment friendly
✔ Extensively documented