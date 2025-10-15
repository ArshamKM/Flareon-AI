# Modelo de Evaluación Flareon-AI

## Caso de Negocio

### Introducción

**Los incendios forestales son fuegos no controlados que se inician y propagan a través de la vegetación silvestre como bosques, pastizales y sabanas.** Son un fenómeno global que afecta los ecosistemas de todos los continentes. Los incendios pueden originarse por causas como rayos o actividades humanas, pero su intensidad y crecimiento pueden estar fuertemente influenciados por condiciones ambientales, como sequías, vientos fuertes y temperaturas elevadas.
Los incendios pueden ocurrir a múltiples niveles:

- **Incendios en el suelo** - en suelos ricos en materia orgánica y raíces
- **Incendios en la superficie** - queman hierba seca, hojas y escombros cerca del suelo del bosque
- **Incendios en la copa** - se propagan rápidamente a través de las copas de árboles y arbustos

A medida que el cambio climático incrementa las sequías y altera los patrones climáticos, los incendios forestales se están volviendo más frecuentes y severos. Esto pone en riesgo a comunidades, ecosistemas y economías.
Además de la amenaza directa a la vida y la propiedad, los incendios pueden degradar la calidad del aire, interrumpir el ecosistema y provocar pérdidas económicas significativas. Por ejemplo, España ha perdido más de **8 millones de hectáreas de su territorio debido a incendios forestales entre 1961 y 2016**, lo que ha resultado en pérdida de biodiversidad, erosión del suelo y miles de millones gastados en lucha contra incendios y esfuerzos de recuperación.

---

### Definición del Problema

España es uno de los países europeos más gravemente afectados por los incendios forestales, tanto ecológica como económicamente. Entre 2011 y 2020, el país enfrentó **219 incendios forestales a gran escala**, que causaron €525 millones en daños. En toda Europa, los incendios han provocado más de **€54 mil millones** en pérdidas económicas entre 2000 y 2017, y se espera que esta cifra aumente a medida que las temperaturas y las sequías continúan incrementándose y crean condiciones para brotes de fuego.
En España, regiones como **Castilla y León y Andalucía** son particularmente vulnerables debido a su clima seco en verano. Los incendios en estas regiones tienen consecuencias ambientales más graves. Además, los incendios pueden afectar negativamente a las poblaciones y dañar la infraestructura.

Los sistemas actuales de gestión y predicción de incendios forestales en España y Europa dependen de **modelos estáticos y simulaciones basadas en datos históricos**. Estos modelos pueden experimentar lagunas de datos y una capacidad predictiva en tiempo real limitada, lo que resulta en una posible **preparación y estrategias de lucha contra incendios insuficientes**.
Dado que los incendios se están convirtiendo en un riesgo sistémico y recurrente, existe la necesidad de una **herramienta de pronóstico avanzada y dinámica que pueda visualizar la posible propagación y gravedad de los incendios**, mientras apoya la mitigación de riesgos.

---

### Solución Propuesta

#### Modelo de Evaluación Flareon-AI (Flareon-AI)

Flareon-AI será un sistema basado en aprendizaje profundo diseñado para:

- Identificar la posible zona donde se propagaría el incendio forestal, basado en **datos históricos de incendios, condiciones climáticas en tiempo real de APIs meteorológicas, que presentan un pronóstico de 16 días**.

Para mayor precisión, nuestro modelo tomará en cuenta **hasta 5 días de los 16 días**, ya que más allá de ese período la precisión disminuye. Por ejemplo, el [Centro Europeo de Previsiones Meteorológicas a Plazo Medio (ECMWF)](https://www.ecmwf.int) es altamente preciso durante los primeros 7-10 días, pero después se vuelve más difícil predecir perfectamente las condiciones climáticas.

- Evaluar su **gravedad y daño potencial**, basado en los conjuntos de datos proporcionados
- Registrar los hallazgos posibles y actualizar el modelo regularmente
- Estimar recursos de lucha contra incendios y estrategias de respuesta para áreas de alto riesgo

*Aspectos que podrían desarrollarse en una etapa posterior:*

- Visualización General:
  - Comparar la temperatura a lo largo de los años, ya sea en España o entre España y otros países - ¿cuál es la razón detrás de esto?
  - Comparar si hay períodos de sequía más altos - ¿la razón es similar a la de la comparación de temperatura o hay otra razón?
  - Si la cantidad de incendios sigue aumentando, ¿cuál sería la posible razón? ¿Son más frecuentes hoy en día o mantienen un flujo constante?
  
- Influencia Vegetal:
  - ¿Cómo influye el tipo de árboles en la propagación, tamaño e intensidad?
  - Visualización: Comparar el tipo de árbol (pino - roble)
  - Uso de imágenes satelitales: ¿Cómo se recupera la vegetación después de un incendio?

#### Conjuntos de Datos
***Estos se han seleccionado como posibles conjuntos de datos confiables, pero existe la posibilidad de cambiarlos más adelante si la visualización no arroja el resultado deseado.***
Para garantizar una visualización y predicciones correctas y precisas, los conjuntos de datos elegidos son:

- **[AEMET - Agencia Estatal de Meteorología](https://www.aemet.es/en/portada)**

Consiste en un gran conjunto de datos abierto, proporcionado por la agencia estatal del Gobierno de España, responsable de ofrecer pronósticos meteorológicos, incluidos **temperatura, velocidad del viento y humedad**.

- **[NASA Prediction Of Worldwide Energy Resources - Data Access Viewer](https://power.larc.nasa.gov/)**

Aplicación web de mapeo receptiva que proporciona herramientas de subconjunto de datos, gráficos y visualización en una interfaz fácil de usar. Se utilizará para recopilar datos agroclimáticos basados en la ubicación de cada provincia del país.

- **[Civio Data](https://datos.civio.es/dataset/todos-los-incendios-forestales/)**

Los datos utilizados en España sobre incendios, tanto en el mapa de incendios como en muchos de los artículos, provienen de las Estadísticas Generales de Incendios Forestales (EGIF), preparadas por el Centro de Coordinación de la Información Nacional sobre Incendios Forestales (CCINIF) basándose en la información anual proporcionada por las comunidades autónomas.
***Fuente: Ministerio de Agricultura, Pesca y Alimentación.***

---

*Durante la investigación de nuestra idea, se encontró que existen modelos similares, como [FLAM](https://iiasa.ac.at/models-tools-data/flam), que capturan los impactos del clima, la población y la disponibilidad de combustible en las áreas quemadas y las emisiones asociadas.*

---

Literatura:

- https://cdnsciencepub.com/doi/full/10.1139/er-2020-0019
- https://www.sciencedirect.com/science/article/pii/S0950705123009486
- https://education.nationalgeographic.org/resource/wildfires/
- https://education.nationalgeographic.org/resource/resource-library-human-impacts-environment/
- https://education.nationalgeographic.org/resource/mapmaker-current-united-states-wildfires-and-perimeters/
- https://www.bsc.es/research-and-development/projects/salus-salus-wildfire-risk-solutions-spain
- https://elobservatoriosocial.fundacionlacaixa.org/en/-/forest-fires-in-spain-importance-diagnosis-and-proposals-for-a-more-sustainable-future#:~:text=The%208%20million%20hectares%20in,due%20to%20loss%20of%20biodiversity%2C
- https://www.ecmwf.int
- https://en.wikipedia.org/wiki/State_Meteorological_Agency
- https://www.london-fire.gov.uk/safety/the-workplace/fire-risk-assessments-your-responsibilities/
- https://www.london-fire.gov.uk/safety/the-workplace/fire-risk-assessments-your-responsibilities/
