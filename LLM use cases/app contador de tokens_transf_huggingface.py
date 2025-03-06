####################################################################################################
# Victor Peral - 2025 
# Python code to calculate the number of tokens of a text using the Hugging Face library
####################################################################################################

from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
text = "Me encanta Python y su aplicacion en Data Science"
tokens = tokenizer.encode(text)
print(f"Número de tokens: {len(tokens)}")
print(f"Tokens: {tokenizer.convert_ids_to_tokens(tokens)}")