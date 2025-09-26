## Exploratory Data Analysis

### Introduction - Anomalies

Wildfires are uncontrolled fires that spread rapidly through wildland vegetation, posing significant threats to ecosystems, human lives, and property. Analyzing wildfire data is essential for understanding their patterns, predicting their behaviour, and developing effective prevention and management strategies. Exploratory Data Analysis (EDA) plays a vital role in this process by allowing us to uncover hidden insights within fire datasets.

Various aspects of wildfires will be investigated through EDA, such as their frequency, burned areas, geographical distribution, and temporal trends. This exploration helps in identifying anomalies, understanding the basic structure of data, and formulate hypotheses for further analysis.
This initial exploration helps in identifying anomalies, understanding the
When discussing wildfires, it's essential to understand the potential reasons behind their occurrence. These can be broadly categorized into natural causes and human-caused factors:

***Natural Causes of Wildfires***
- **Lightning Strikes**: This is the most common natural cause of wildfires. Both "cold" lightning (short duration, intense current) and "hot" lightning (less voltage but longer duration) can ignite dry vegetation, especially hot lightning.
- **Volcanic Eruptions**: Lava and ash from volcanic activity can generate enough heat to start fires.
- **Spontaneous Combustion**: In rare cases, decomposing organic materials like large piles of dry leaves or compost can generate enough heat to spontaneously ignite.

***Human-Caused Wildfires***

It's important to note that a significant majority of wildland fires are caused by human activities. These can result from accidental negligence or intentional acts:

- **Unattended Campfires**: Campfires that are left unsupervised or not properly extinguished are a leading cause of wildfires.
- **Burning Debris**: Carelessly burning yard waste, agricultural debris, or slash piles can easily lead to uncontrolled fires, especially in hot, dry, or windy conditions.
- **Equipment Use and Malfunctions**: Sparks from vehicles (e.g., hot exhaust pipes, dragging chains), machinery (e.g., chainsaws, grinders), and electrical infrastructure (e.g., downed power lines, faulty cables) can ignite dry vegetation.
- **Discarded Cigarettes**: Negligently discarded cigarettes can easily ignite dry grasses or brush.
- **Fireworks**: The use of fireworks, especially in dry conditions, can easily spark wildfires.

***Beyond specific ignition sources, several environmental factors exacerbate wildfire risk:***
- **Weather Conditions**: High temperatures, low humidity, strong winds, and prolonged droughts create ideal conditions for fires to start and spread rapidly. Wind, in particular, can pre-heat fuels and carry embers, causing spot fires ahead of the main blaze.
- **Climate Change**: A significant contributor to worsening wildfire seasons, climate change leads to hotter, drier conditions, extended fire seasons, and increased frequency and severity of extreme weather events like droughts and lightning storms, making environments more susceptible to burning.

Understanding the complex interactions among human activities, natural phenomena, and environmental conditions could result in better and more effective prevention and management.

### Wildfires

Wildfires, as devastating and uncontrolled blazes, represent a critical environmental challenge with far-reaching consequences for ecosystems, human settlements, and even atmospheric composition.
By systematically investigating fire data, we can gain a clearer picture of various aspects, such as:
- **Frequency and Distribution**: How often do wildfires occur, and where are they most prevalent?
- **Burned Areas and Severity**: Analyzing the extent of land consumed by fires helps us understand the scale of destruction and identify areas most vulnerable to severe impacts. This involves examining metrics like the amount of burned area.
- **Temporal Trends**: Are wildfires becoming more frequent or intense over time?

***Dataset***

- **Observation**: This dataset contains recorded wildfires that occurred across Spain from 2001 until 2021, with detailed data about each fire's location, impact and response.
- **Column & Variables**:

| Column Name            | Data Type | Description                                                            |
|------------------------|-----------|------------------------------------------------------------------------|
| `fire_id`              | int64     | Fire Identifier                                                        |
| `burned_area_ha`       | float64   | Forest area burnt in hectares                                          |
| `date`                 | object    | Date of detection of the fire (format yyyy-mm-dd)                      |
| `latitude`             | float64   | Geographical latitude of fire origin                                   |
| `longitude`            | float64   | Geographical longitude of fire origin                                  |
| `region_id`            | int64     | Identifier of the region                                               |
| `province_id`          | int64     | Identifier of the province                                             |
| `municipality_id`      | int64     | Identifier of the municipality                                         |
| `municipality`         | object    | Name of municipality                                                   |
| `cause_id`             | int64     | Identifier of the cause of the fire                                    |
| `deaths`               | int64     | Number of the deaths in the fire                                       |
| `time_to_control_m`    | int64     | Elapsed time to enter fire control phase (in minutes)                  |
| `time_to_extinguish_m` | int64     | elapsed time to extinguish the fire (in minutes)                       |
| `extinction_cost`      | int64     | Extinguishing costs associated with the fire as reported in EGIF       |
| `economic_loss`        | int64     | Economic losses associated with the fire as reported in EGIF (in euro) |

- **Key Features**:
  - ***Geospatial Coverage***: Fires span 50 provinces (e.g., 33 = Asturias, 39 = Cantabria).
  - ***Temporal Coverage***: January to December in the period 2001 - 2021.
  - ***Impact Metrics***: Includes area burned, response times, and costs.
  - ***Cause Analysis***: Differentiates between negligent and intentional causes.
  - ***Sample Data***:
- **Limitations:**
  - ***Time range***: Only covers the period of 2001 - 2021 (incomplete the last 4 years)
  - ***Cause Granularity***: `cause_id` lacks descriptive labels (not defined types)
- **Correlations**:
  - ***Strongest relationships (Best)***:
1. `time_to_extinguish_m` and `time_to_control_m` **(Correlation: 0.67)**: as the time to control fire increases, the time to extinguish it also leans to increase significantly.
2. `economic_loss` and `burned_area_ha` **(Correlation: 0.39)**: larger burned areas are associated with higher economic losses.
3. `burned_area_ha` and `time_to_extinguish_m` **(Correlation: 0.29)**: larger burned areas leans to take a bit longer to extinguish.
  - ***Weakest relationships (Worst: closes to zero)***:
1. `deaths` and `time_to_control_m` **(Correlation: 0.028)**: there is almost no linear association between the number of deaths and the time to control the fire.
2. `deaths` and `extinction_cost` **(Correlation: 0.014)**: the number of deaths has almost no linear association with the extinction cost.
3. `burned_area_ha` and `extincion_cost` **(Correlation: 0.09)**: the size of the burned area has a minimal linear association with the extinction cost.

***The weak correlations don't mean there's no relationship at all, just that there isn't a strong linear relationship between those specific variables.*** (see the graph below)

#### Section Graphs here

### Wildfires - Conclusion
Wildfire data shows that most fires are relatively small, typically covering less than 10 hectares, and fortunately, few result in fatalities. However, these incidents still pose significant threats, with some municipalities facing considerable economic losses. The costs associated with extinguishing fires are complex and influenced by factors beyond just the size of the burned area, suggesting that terrain and accessibility to infrastructure play critical roles.

Furthermore, wildfire occurrences exhibit distinct seasonal and geographical patterns, with activity peaking in March, July, and August. Certain provinces, such as Asturias and Ourense, experience a higher number of fires. While the annual number of wildfires has varied over the years, the data indicates a decline in recent years, suggesting a potential change in fire frequency.

In the next phase of the investigation, we will focus on identifying the specific weather conditions, beyond just temperature, that may serve as catalysts for wildfires to ignite and spread. By examining factors such as temperature and precipitation patterns, we aim to uncover the relationships that ultimately determine the likelihood of wildfire outbreaks.

### Introduction - Possibilities

Weather plays a crucial role in shaping our environment, and its intricate relationship with wildfires is particularly important.

To fully understand the dynamics of wildfires, we need to explore a wide range of weather conditions. Our upcoming analysis will examine a weather dataset to uncover the specific influence of various meteorological factors, including patterns of precipitation (or the absence of it) and other atmospheric variables. By closely investigating these parameters, we aim to gain extended insights into how weather acts as a key determinant in the occurrence and behaviour of these devastating natural events.

### Weather

By carefully examining the weather dataset, we aim to gain a clearer understanding of various aspects, including:
- **Temperature's Impact**: What are the average, minimum, and maximum daily temperatures immediately preceding wildfire incidents? Are there specific temperature thresholds above which the probability of a wildfire significantly increases?
- **Precipitation Mitigating**: How long do dry spells (periods without significant rainfall) typically persist before a wildfire occurs, and at what point does the risk become critical?
- **Meteorological Effects**: What precise combination of high temperatures and low precipitation creates the most hazardous conditions for wildfire initiation?

***Dataset***

- **Observation**: A daily weather dataset, containing geographical details and temperature/precipitation readings from stations to unveil local climatic patterns and anomalies.
- **Column & Variables**:

| Column Name            | Data Type | Description                                                                                 |
|------------------------|-----------|---------------------------------------------------------------------------------------------|
| `STATION`              | object    | Unique identifier for the weather station                                                   |
| `NAME`                 | object    | Name of the weather station                                                                 |
| `LATITUDE`             | float64   | Geographical latitude                                                                       |
| `LONGITUDE`            | float64   | Geographical longitude                                                                      |
| `ELEVATION`            | float64   | The altitude or height of the weather station above sea level, typically measured in meters |
| `DATE`                 | object    | Date of recorded observations (format yyyy-mm-dd)                                           |
| `PRCP`                 | float64   | Precipitation amount for the day (in millimeters)                                           |
| `TAVG`                 | float64   | Average temperature for the day (in celsius)                                                |
| `TMAX`                 | float64   | Maximum temperature for the day (in celsius)                                                |
| `TMIN`                 | float64   | Minimum temperature for the day (in celsius)                                                |

- **Key Features**:
  - ***Geospatial Specificity:*** Data is tied to different stations with precise latitude, longitude and elevation
  - ***Daily Granularity:*** Observations are recorded on a daily basis
  - ***Quantitative Measurements:*** All-weather related columns are numerical, allowing for easier analysis and visualization.
  - ***Sample Data***:
- **Limitations:**
  - ***Time range***: Only covers the period of 2001 - 2021 (incomplete the last 4 years).
  - ***Humidity and Wind Values***: Currently, the dataset does not have humidity and wind, but in the future, they could be implemented.
- **Correlations**:
  - ***Strongest relationships (Best)***:
1. `TMAX` and `TAVG` **(Correlation: 0.96)**
2. `TAVG` and `TMIN` **(Correlation: 0.94)**
3. `TMAX` and `TMIN` **(Correlation: 0.84)**
   - ***Weakest relationships (Worst: closes to zero)***:
1. `PRCP` and `TMIN` **(Correlation: -0.057)**
2. `ELEVATION` and `TMIN` **(Correlation: -0.46)**
3. `ELEVATION` and `TMAX` **(Correlation: -0.41)**

***The weak correlations don't mean there's no relationship at all, just that there isn't a strong linear relationship between those specific variables.*** (see the graph below)

#### Section Graphs here

### Weather - Conclusion

Analyzing climate data from 2001 to 2021 reveals significant year-to-year fluctuations in both average temperatures and yearly precipitation, rather than a clear long-term trend. Average temperatures ranged from a low of 13.63°C in 2008 to a high of 16.12°C in 2004, indicating considerable variability over the two decades. Similarly, average yearly precipitation has shown substantial shifts, with a minimum of 1.34 mm in 2004 and a maximum of 2.42 mm in 2018, further highlighting the absence of a consistent upward or downward pattern.

The distribution of average temperatures exhibits a bimodal characteristic, suggesting the presence of two dominant temperature ranges or seasons. This bimodal pattern, with peaks indicating more frequent occurrences of temperatures within specific cool and warm ranges (approximately between 5°C and 25°C), reflects the nature of temperature changes throughout the year. In essence, the region's climate over this period is marked by distinct seasonal shifts and notable year-to-year variability in both temperature and precipitation.

### Conclusion 

The analysis of wildfire and weather data shows a strong link between climate and wildfire activity. There are two main temperature peaks during the year, indicating clear warm and cool seasons. Wildfires often occur most in the warm months of July and August, with some activity in March. Higher temperatures and drier conditions lead to more wildfires, especially when there’s less rainfall, which dries out vegetation and makes it easier for fires to start.

The changes in temperature and rainfall directly affect the patterns of wildfires each season and from year to year. Warm and dry periods make vegetation more flammable, leading to more fires, especially in certain areas and seasons. On the other hand, more rain can reduce fire activity by keeping fuels moist. While most wildfires are small, they can have significant economic and environmental impacts, highlighting the importance of specific weather conditions on wildfire risk.

**Next Steps**
- Test different models, who can work well together with the dataset (ex. a predictive model, such as a regression or ensemble method)
- Select and visualize to reveal geographic and seasonal patterns in fire costs, which would provide qualitative context, deepening the understanding an support evidence for future work.