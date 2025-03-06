##################################################################################################
# Victor Peral - 2025
#
# This code performs the sentiment analyses of a given text using the Hugging Face Transformers library. 
# ####################################################################################################

from transformers import pipeline

# Inicializar el modelo de análisis de sentimientos
sentiment_analyzer = pipeline('sentiment-analysis')

# Definir el texto a analizar
text = "Me encanta programar en Python, es muy divertido y útil."

# Realizar el análisis de sentimientos
result = sentiment_analyzer(text)

# Imprimir el resultado
print(f"Sentimiento: {result[0]['label']}")
print(f"Puntuación: {result[0]['score']:.2f}")