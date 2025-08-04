# Chronic Diseases Predictor With Chatbot

## Overview
This project is a web application that uses machine learning to predict chronic diseases. It helps healthcare providers make faster and more accurate diagnoses by analyzing patient data. The system also includes a chatbot for general health information. The main goal is to promote preventive care by identifying health risks early.

## Key Features
**Multi-Disease Prediction:** The application can predict the likelihood of several chronic diseases, including:
* `Diabetes`
* `Heart Disease`
* `Kidney Disease`
* `Liver Disease`

**Medical Chatbot:** An interactive chatbot is integrated to provide users with general health information, answer questions about symptoms, and explain diseases.

## Disease Prediction Implementation
The disease prediction process is handled by a series of steps:
* **Pre-trained Models:** The backend loads pre-trained machine learning models for each specific disease using Python's pickle library.
* **User Input:** A user enters their health data into a form on the web application.
* **Data Processing:** This input is passed to the corresponding disease prediction model.
* **Output:** The model returns a numerical value.
* **Result Display:** The application translates the numerical output into a percentage, which represents the user's chance of having the disease. This result is then displayed to the user.

## Chatbot Implementation
The medical chatbot in this project was built using the RAG (Retrieval-Augmented Generation) method, following these key steps:
* **Knowledge Base:** Information from a 638-page medical book was embedded and stored in a Vector Database.
* **User Query:** When a user asks a question, the system searches the Vector Database for the most relevant information.
* **LLM Processing:** This retrieved information is then sent as context to a Large Language Model (LLM) using an API key from GROQ Cloud.
* **Response Generation:** The LLM processes the context to generate a simple, clear, and relevant reply.
* **Domain Guardrails:** The chatbot is configured to only answer medical-related questions and will politely decline if asked about unrelated topics.

## Technologies Used
* **Backend:** `Python`
* **Web Framework:** `Flask`
* **Machine Learning:** `scikit-learn`, `tensorflow.keras`, `numpy`, `pickle`
* **Frontend:** `HTML`, `CSS`
* **Server:** `Flask` Development Server

## Screenshots
### Home Page:
<img width="900" height="900" alt="Home_page" src="https://github.com/user-attachments/assets/395472b9-7a87-406e-a976-ff8f26a44b98" />

### Diabetes Predictor:
<img width="900" height="900" alt="diabetes_predict" src="https://github.com/user-attachments/assets/e36efcc3-647b-4ea7-8e5c-4d3226affa8e" />
<img width="900" height="800" alt="diabetes_output" src="https://github.com/user-attachments/assets/80532944-6c2c-470d-bd30-ffb40b0377ef" />

### Medical Chatbot:
<img width="900" height="900" alt="chatbot" src="https://github.com/user-attachments/assets/a77963fd-8564-4522-b093-ece13788520d" />







