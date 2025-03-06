####################################################################################################
# Victor Peral - 2025 
#
# RAG application. The code uses two different models to implement the RAG system:
# For the calculation of embeddings: The model used is 'paraphrase-MiniLM-L6-v2', which is a variant of the all-MiniLM-L6-v2 model. 
# This model maps sentences and paragraphs to a dense 384-dimensional vector space1. 
# It is used to encode both the knowledge base and the user's query into vectors, allowing similarity search to retrieve the relevant context.
# For the generation of the response: The model used is 'google/flan-t5-base', which is a larger version of the T5 (
# Text-to-Text Transfer Transformer) model developed by Google. This model is used to generate the final response based on the retrieved context and the user's question.
# ####################################################################################################

from sentence_transformers import SentenceTransformer
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Base de conocimiento ampliada
knowledge_base = [
    "RAG significa Generación Aumentada por Recuperación.",
    "RAG se usa para mejorar la precisión y relevancia de las respuestas de los LLM.",
    "RAG combina la recuperación de información con la generación de texto.",
    "RAG permite a los modelos de lenguaje acceder a información externa durante la generación de respuestas.",
    "RAG mejora la capacidad de los LLM para proporcionar respuestas actualizadas y precisas.",
    "En RAG, se recupera información relevante de una base de conocimientos antes de generar una respuesta.",
    "RAG ayuda a reducir las alucinaciones en los modelos de lenguaje al proporcionar contexto factual.",
    "RAG es especialmente útil en tareas que requieren conocimientos específicos o actualizados."
]

# Cargar modelo de embeddings
embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Generar embeddings para la base de conocimiento
knowledge_embeddings = embedding_model.encode(knowledge_base)

# Cargar modelo de generación de texto
model_name = "google/flan-t5-base"  # Cambiamos a un modelo más grande
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def retrieve_context(query, top_k=5):  # Aumentamos el número de contextos recuperados
    query_embedding = embedding_model.encode([query])
    similarities = np.dot(query_embedding, knowledge_embeddings.T)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    return [knowledge_base[i] for i in top_indices]

def generate_response(query, context):
    prompt = f"""
    Tarea: Responde a la siguiente pregunta de manera detallada y precisa, utilizando la información proporcionada en el contexto. Asegúrate de incluir una definición clara, explicar su funcionamiento y mencionar sus principales beneficios.

    Contexto:
    {' '.join(context)}

    Pregunta: {query}

    Respuesta detallada:
    """
    
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(**inputs, max_length=300, num_return_sequences=1, temperature=0.7, do_sample=True, top_k=50, top_p=0.95)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Post-procesamiento para mejorar la respuesta
    response = response.replace("Respuesta detallada:", "").strip()
    if not response.startswith("RAG"):
        response = "RAG " + response
    
    return response

# Ejemplo de uso
query = "¿Qué es RAG y para qué se usa?"
context = retrieve_context(query)
response = generate_response(query, context)
print(f"Contexto: {context}")
print(f"Pregunta: {query}")
print(f"Respuesta: {response}")
