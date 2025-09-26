# WildFires Assessment Model

## Business Case

### Introduction

**Wildfires are uncontrolled fires that ignite and spread across wild land vegetation such as forests, grasslands, and savannas.** They are a global phenomenon, impacting the ecosystems of every continent. Wildfires can emerge from causes like lightning or human activities, but their intensity and growth can be greatly influenced by environmental conditions—drought, high winds, and elevated temperatures being key factors.
The fires can occur at multiple levels:

- **Ground fires** - in organic-rich soil and roots
- **Surface fires** - burn by dry grass, leaves and debris close to the forest floor
- **Crown fires** - spread rapidly through the canopies of trees and shrubs

As climate change increases droughts and changes weather patterns, wildfires are becoming more frequent and severe. This places communities, ecosystems and economies at high risk.
Beyond the direct threat to life and property, wildfires can degrade air quality, disrupt the ecosystem and lead to significant economic losses. For example, Spain has had over **8 million hectares of its land ruined by wildfires between 1961 and 2016**, resulting in biodiversity loss, soil erosion and billions spent on firefighting and recovery efforts.

---

### Problem Definition

Spain is one of the European countries most severely impacted by wildfires, both ecologically and economically. Between 2011 and 2020, the country faced **219 large-scale wildfires**, resulting in €525 million in damages. Across Europe, wildfires have caused over **€54 billion** in economic losses between 2000 and 2017, and it's expected to rise as temperatures and droughts continue to rise and create conditions for fire outbreaks. 
In Spain, regions like **Castilla y León and Andalucía** are particularly vulnerable due to their dry summer climate. The wildfires in the regions lead to more serious environmental consequences. Moreover, wildfires could affect negatively populations and damage infrastructure.

Current wildfire management and forecasting systems in Spain and Europe rely on **static models and historical data-based simulations**. These models could experience data gaps and limited real-time predictive capacity, resulting in possible **insufficient preparation and firefighting strategies**.
With wildfires becoming a systemic and recurring risk, there is a need for **advanced, dynamic forecasting tool that can visualize potential fire spread and severity**, while supporting risk mitigation.

---

### Proposed Solution

#### WildFire Assessment Model (WFAM)

WFAM will be a deep-learning based system designed to:

- Identify the possible zone, where the wildfire would spread, based on **historical wildfire data, real-time weather conditions from Weather APIs, which are presenting a 16-day forecast**.

For better accuracy, our model will take into account **10 days out of 16 days**, since further into the future they become less accurate. For example, the [European Centre for Medium-Range Weather Forecasts (ECMWF)](https://www.ecmwf.int) is highly accurate for the first 7-10 days, after that, it becomes more difficult to predict the weather conditions perfectly.

- Assess its **severity and potential damage**, based on the provided datasets
- Record the possible findings and update the model regularly
- Estimate firefighting resources and response strategies for high-risk areas

*Things that could be developed in later stage:*

- General Visualization:
  - Comparing the temperature through the years, either in Spain or between Spain and other countries - what is the reason behind it?
  - Comparing if there is higher drought periods - is the reason similar to the one when comparing temperature or there is another reason?
  - If the amount of fires keep rising, what would be the possible reason? Are they more nowadays or maintain a steady flow?
  
- Vegetational Influence:
  - How does the type of trees could influence the spread, size and intensity?
  - Visualization: Comparing the tree type (pine - oak)
  - Use of satellite pictures: How does the vegetation recover after a fire?

#### Datasets
***These are chosen as a possible reliable datasets, but there is a possibility to be changed later, if the visualization does not return the desired outcome.***
To ensure correct and accurate visualization and predictions, the datasets chosen are:

- **[AEMET - State Meteorological Agency](https://www.aemet.es/en/portada)**

Consists of a large open dataset, given by state agency of the Government of Spain, responsible for providing weather forecasts, including in **temperature, wind speed and humidity**.

- **[NASA Prediction Of Worldwide Energy Resources - Data Access Viewer](https://power.larc.nasa.gov/)**

Responsive web mapping application providing data subsetting, charting, and visualization tools in an easy-to-use interface. It will be used to gather agroclimatology data based on the location of each province in the country.  

- **[Civio Data](https://datos.civio.es/dataset/todos-los-incendios-forestales/)**

The data used in Spain on fire, both on the fire map and in many of the articles, comes from the General Statistics on Forest Fires (EGIF), prepared by the National Forest Fire Information Coordination Center (CCINIF) based on the annual information provided by the autonomous communities.
***Source: Ministry of Agriculture, Fisheries, and Food.***
---

*During the research of our idea, it was found that there are similar models, like [FLAM](https://iiasa.ac.at/models-tools-data/flam), which capture impacts of climate, population, and fuel availability on burned areas and associated emissions.*

---

Literature:

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
