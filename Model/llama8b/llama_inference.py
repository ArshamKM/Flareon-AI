from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Nombre exacto del modelo en Hugging Face
MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"

# Carga de tokenizer y modelo (usará tu token de Hugging Face guardado)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto"  # aprovecha GPU/Metal si está disponible
)

# Pipeline de generación de texto
llm = pipeline("text-generation", model=model, tokenizer=tokenizer)

def infer(prompt: str) -> str:
    """
    Envía un prompt a LLaMA y devuelve el texto generado.
    """
    output = llm(prompt, max_new_tokens=300)
    return output[0]["generated_text"]