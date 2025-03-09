# My AI/ML and GenAI projects in Python

# Just a few words... 
Welcome to my GitHub repository! Here you will find several projects on Python I have been working on for some time, that helped me connect some key principles and fundamental Advanced Analytics & Data Science concepts, with special focus on ML and LLM.  
In my way to explore the Data Science & AI field, I really believe it is important to play, test, build, make mistakes, learn and have fun! #Never_Stop_Learning 🚀😊

# 1. multi-ML_comparison_Flask_frontend

<b> multi-ML_comparison_Flask_frontend.py</b> is a multi-ML choice application, it uses  http://api.worldbank.org public data for model training. Server and Frontend code: index.html shows real and predicted values, and compares the models performance statistics. The application plots 3d graphs created with Plotly that are shown in a web app created with Flask .

In this project I'm using modules such as Pandas, Numpy, Plotly, scikit-learn, Flask, JSON, etc. to analyze the relationship between data from databases and calculate predictions using models such as linear regression, assembly methods, etc. It display the graphs in a Web app created in Flask.

# 2. LLM use cases projects

2.1. <b> app contador de tokens_transf_huggingface.py:</b> 

<b> Tokens counter </b>. Python code to  calculate the number of tokens of a text using the Hugging Face library

2.2. <b> app RAG_transformer_huggingface.py:</b>

<b> RAG application </b> . The code uses two different models to implement the RAG system:

For the calculation of embeddings: The model used is 'paraphrase-MiniLM-L6-v2', which is a variant of the all-MiniLM-L6-v2 model. This model maps sentences and paragraphs to a vector space. 
It is used to encode both the knowledge base and the user's query into vectors, allowing similarity search to retrieve the relevant context.

For the generation of the response: The model used is 'google/flan-t5-base', which is a larger version of the T5 (
Text-to-Text Transfer Transformer) model developed by Google. This model is used to generate the final response based on the retrieved context and the user's question.

2.3. <b>app sentiment analysis_transf_huggingface.py:</b>

 <b> Sentiment analysis </b>. This code performs the sentiment analysis of a given text using the Hugging Face Transformers library. 

2.4. </b>app text summary_transformers_huggingface.py:</b>

 <b> Text summarization </b>. This code performs text summarization of a given text using the Hugging Face Transformers library. 

# 3. ML-based price prediction in three simulation scenarios of demand/competition

ML price prediction_demand_competition scenarios.py: ML application to predict the price given 3 scenarios of competition (severe, mid,low). index.html is generated to plot the 3d graphs that shows the predicted price evolution.
It uses Linear Regression model, the graphs are created with Plotly and the server-web FrontEnd with Flask.

# 4. ML use case in Pharma: drug efficacy prediction from a random clinical trial data

<b>ML_drug_efficacy prediction_clinical_study_SQL_randomDB.py:</b> ML model for clinical study - prediction of the drug efficacy based on the dose and age. Usage of a linear regression model.

<b> WARNING: the model is trained with a random dataset of clinical study created just for the purpose of validating the end-to-end process of ML model creation </b>
