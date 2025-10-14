def build_prompt(results):
    """
    Crea un prompt de texto para el modelo LLaMA a partir de los
    resultados de predicción de incendios.

    results: normalmente un DataFrame de pandas o un dict.
    """
    # Si es un DataFrame, hacemos un resumen rápido
    try:
        summary = results.head(10).to_csv(index=False)
    except AttributeError:
        summary = str(results)

    prompt = (
        "Analiza el siguiente resumen de predicciones de riesgo de incendio:\n"
        f"{summary}\n"
        "Devuelve un breve análisis en lenguaje natural destacando las provincias con mayor riesgo "
        "y posibles recomendaciones de prevención."
    )
    return prompt