# Prevention of Wildfires Using a Wildfire Assessment Model - Project Charter

## Project Name: WildFire Assessment Model (WFAM)
#### Latest Version Date: 08/04/2025

### Project Purpose
The primary goal of WFAM is to improve wildfire prevention and enhance the monitoring of vulnerable areas in Spain. By developing a predictive machine learning model, the project aims to provide early detection of high-risk zones, allowing organizations to take proactive measures in mitigating wildfire damage.

### Project Objective
The WFAM objective will be to develop a learning model that reads historical wildfire data and real-time weather inputs to identify and visualize a easily ignitable areas, aiding for risk-mitigation and decision-making.

### SMART Goals
Develop a Predictive model with minimum 80% Accuracy:
- ***Specific***: Build a machine learning model to predict high-risk wildfire zones.
- ***Measurable***: Achieve at least 80% accuracy on validation data.
- ***Achievable***: Based on previous studies and available data, 80% is a realistic benchmark.
- ***Relevant***: Model accuracy directly affects the reliability of early warnings.
- ***Time-bound***: Goal to be reached by Week 12 of the project
  
### General Project Information
WFAM will be a deep-learning-based system designed to identify the possible zone / zones, where the wildfire would spread, based on historical wildfire data and real-time weather conditions from Weather APIs, which are using a 10-day forecast for a maximum accuracy. The model will provide early warnings to aid in wildfire prevention and resource allocation.
- The project itself was pitched on 01/04/2025 and based on the feedback recieved, the project idea recieved mostly a positive feedback, regarding it's business case and what will cover.

## **Scope**  
A brief summary of what will and will not be included in the project tasks and prevent misunderstandings
### **Inclusions**  
- Development of a machine learning model trained on:  
  - Historical wildfire data (location, severity, causes).  
  - Real-time weather data (temperature, humidity, wind).  
- Identification of high-risk zones.
- Integration of model outputs into a visualization dashboard

### **Exclusions**  
- On-the-ground firefighting operations.  
- Hardware deployment (e.g., sensors, drones).
- Legal and policy recommendations

## **Deliverables**  
### **Group Deliverables**  
- [Project Charter](https://github.com/FontysVenlo/PRJ4-2025-prj4-2025-d07/blob/main/Project%20Charter.md)
- [Business Case](https://github.com/FontysVenlo/PRJ4-2025-prj4-2025-d07/blob/63586ea964900f6573e77ea66969f214209d034e/Business%20Case.md)
- [GitHub Repository](https://github.com/FontysVenlo/PRJ4-2025-prj4-2025-d07/tree/main)

### **Individual Deliverables**  
- Personal Development Plan (skills tracking).  
- Individual GitHub Repository (task-specific contributions).  

### **Risks**
During the documentation and visualization of the deep-learning model, certain risks may occur:
  - Adding unplanned changes in the specification or expanding requirements with more features could lead to missing deadlines.
  - Incomplete, outdated or inconsistent wildfire and weather data may affect model accuracy.
  - Historical records may not account for new wildfire patterns caused by climate change.
  - The model may not achieve the desired accuracy, leading to unreliable predictions.
  - If during the visualization period, the selected dataset is modified or removed, adjustments will be needed, which would lead to missing deadlines and redoing all changes done until this moment.

## **Resources**
- Team consistency: A team of three students will work on the model and have a feedback meeting with the lecturers, who provide comments on the team's work.
- Time: 4 months for project completion
- Budget: (the bidget will be calculated based on the time the team spends on the AI model)
  The team meets for 3 hours each week at the university. However, during project weeks, the team dedicates a total of 20 hours per week, working 4 hours each day. The overall expected effort amounts to 280 hours, although it may exceed this total, as each team member is expected to work at least 4 or more additional hours outside of the scheduled meetings.
- Technical tools and frameworks:
  - Programming languages:
    - Python: for data processing, model development and integration.
  - Libraries & Frameworks:
    - TensorFlow / PyTorch: For building and training the deep learning model.
    - Scikit-learn: For classical machine learning algorithms and evaluation metrics.
    - Pandas / NumPy: For handling structured data and numerical computations.
    - Matplotlib / Seaborn / Plotly: For visualization and analysis of wildfire patterns.
- Computer resources:
  - Local Machines (Developer Laptops/Desktops): used for processing, data exploration, and testing smaller model iterations.
  - Python Libraries & Frameworks:
    - Sckit-learn: for data analysis and preprocessing tasks;
    - Pandas / NymPy: for data manipulation and numerical operations;
    - Matplotlib / Seaborn / Plotly: for data visualization;

  - Data Storage and Access:
    - GitHub: For storing datasets, model checkpoints and documentation;
    - Datasets (e.g. Kaggle and EFFIS) and API's (e.g., AEMET): real=time and historical data retrieval;

### Datasets
***These are chosen as a possible reliable datasets, but there is a possibility to be changed later, if the visualization does not return the desired outcome.***
To ensure correct and accurate visualization and predictions, the datasets chosen are:

- **[AEMET - State Meteorological Agency](https://www.aemet.es/en/portada)**

Consists of a large open dataset, given by state agency of the Government of Spain, responsible for providing weather forecasts, including in **temperature, wind speed and humidity**.

- **[NASA Prediction Of Worldwide Energy Resources - Data Access Viewer](https://power.larc.nasa.gov/)**

Responsive web mapping application providing data subsetting, charting, and visualization tools in an easy-to-use interface. It will be used to gather agroclimatology data based on the location of each province in the country.  

- **[Civio Data](https://datos.civio.es/dataset/todos-los-incendios-forestales/)**

The data used in Spain on fire, both on the fire map and in many of the articles, comes from the General Statistics on Forest Fires (EGIF), prepared by the National Forest Fire Information Coordination Center (CCINIF) based on the annual information provided by the autonomous communities.
***Source: Ministry of Agriculture, Fisheries, and Food.***

***(Regarding the datasets, the team has the ability to modify them at any time. Therefore, when changes occur, the team will update the business case and the project charter accordingly to ensure the documentation remains current.)***

## Quality Management
To maintain high-quality code and deliverables, the team will follow structured development, and clearly defined standards for completion:
- Code Reviews & Vision Control
  - All code will be commited through **GitHub**, using branches for features and tasks;
  - Code must follow consistent formatting, naming conventions and documentation guidelines
  - All work will be reviewed by at least one other team member before uploading or presenting it in front of the lecturers
- Testing & Validation
  - The model will be evaluated using standard metrics (accuracy and precision) and cross-validation
  - Visual outputs (graphs and dashboards) will be tested for usability and clarity
- Agile Workflow & Team Communication
The team will follow **Agile methodologies** using Scum-based approach.
- A Scrum board will be used to manage tasks across columns such as:
  - Todo
  - In progress
  - In review
  - Done
- **Daily Standup Meetings (15 minutes) will be held to:
  - Share what each member did yesterday
  - Identify what will they do today
  - Highlight any challenges

***Definition Of Done (DoD)***

Each task will be considered **complete** only when it meets the following criteria:
- The code is reviewed, merged and free of crucial bugs
- Functionality is verified against requirements
- Documentation is written or updated (e.g, comments, README, usage instructions)

***Weakly team syncs will be used to review progress, align priorities, and discuss quality improvements.***


***A possible version of the Timeline with the work to be done after the first two project weeks is presented [here](Timeline_0704.png).*** 
It's possible to go under change, depending on the work completed during the sprints.

## **Timeline**  
- **Duration**: 14 weeks.  
- **Completion Date**: June 2025.   

### Project Milestones 
  - Week 1-3: Project Initial Idea and Data collection
  - Week 4-8: Data cleaning & visualization
  - Week 9-12: Model development
  - Week 13-14: Final review & documentation

## **Project Team Structure**
- Arsham Khoshfeiz Maghanaki (responsibility)
- Daniel Schmidt (responsibility)
- Hristina Pavlova (responsibility)

## **Approval**  
- **Lecturers**:
  - Mr. Ralf Raumanns
  - Mr. Geert Monsieur
    
Date of approval: (Not yet decided)

---  
References:
- https://plaky.com/learn/project-management/project-charter/
- https://asana.com/resources/project-charter
