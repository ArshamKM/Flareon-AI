# Wildfire Dataset Documentation
#### Part One - Wildfire Documentation

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

#### Part Two - Weather Documentation
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
