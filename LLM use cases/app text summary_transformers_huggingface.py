##################################################################################################
# Victor Peral - 2025
#
# This code performs text summarization of a given text using the Hugging Face Transformers library. 
# ####################################################################################################

from transformers import pipeline

# Inicializar el modelo de resumen
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Texto de ejemplo
texto_largo = """
Python es un lenguaje de programación versátil y poderoso que se ha vuelto 
extremadamente popular en los últimos años. Fue creado por Guido van Rossum 
y lanzado por primera vez en 1991. Python es conocido por su sintaxis clara 
y legible, lo que lo hace ideal tanto para principiantes como para programadores 
experimentados. Se utiliza en una amplia gama de aplicaciones, desde desarrollo 
web y análisis de datos hasta inteligencia artificial y aprendizaje automático. 
Python tiene una gran comunidad de desarrolladores y una extensa biblioteca de 
paquetes que facilitan la realización de tareas complejas con pocas líneas de código.
"""

# Generar el resumen
resumen = summarizer(texto_largo, max_length=100, min_length=30, do_sample=False)

print("#########################")
print(resumen)
# Imprimir el resumen
print("Resumen generado:")
print(resumen[0]['summary_text'])

# Comparación de longitudes
print(f"\nLongitud del texto original: {len(texto_largo.split())}")
print(f"Longitud del resumen: {len(resumen[0]['summary_text'].split())}")