# AI-Driven Assessment of Trends in Climate Policy

## Quick navigation
[Background](#background)  
[Data](#data)  
[Models](#models)  
[Timeline](#timeline)  
[Repo Structure](#repo-structure)  
[Logistics](#project-logistics)  
[Resources](#resources)  
[Contact](#contact-info)

## Goal

The goal of this project is to develop an AI-powered question-answering system that automatically analyzes Climate Action Plans (CAPs) and other climate adaptation and mitigation documentation. The system will be capable of extracting key data about climate vulnerabilities, planned mitigation measures, and socio-economic and geographic context, providing well-sourced, accurate responses to user queries. 

## Background 

Climate change poses an urgent challenge for cities worldwide, prompting the creation of comprehensive Climate Action Plans (CAPs) to mitigate impacts and adapt to evolving conditions. These plans detail strategies for reducing emissions, addressing vulnerabilities, and protecting populations from climate risks, but their length and complexity make it difficult for city planners, researchers, and policymakers to efficiently extract and compare key information across regions.

This project addresses this by developing an AI-powered question-answering system that automates the extraction of critical information from CAPs. Using Natural Language Processing (NLP) and Machine Learning (ML) techniques, the system analyzes thousands of pages of climate documentation and provides accurate, well-sourced responses to climate-related inquiries, with LangChain facilitating the organization and structuring of extracted data for more efficient analysis.

## Data

Climate action Plans can be found under the CAPS folder. External data sources are housed on [Box](https://vanderbilt.box.com/s/g0uz2xpp6eawnmn2704cbpn6gf98vvol)

## Timeline

Fall 2024 (September through December 2024 intially)

## Repo Structure 

This repository contains code for three main components: 

1. Data ingestion and processing portal
2. Climate Action Plan QA Tool
3. Climate Action Plan Maps Tool

Both the QA and Maps tools are hosted on Streamlit Cloud as well as HuggingFace Spaces. They may also be run locally. The data ingestion and processing portal is designed to be run locally. 

Users can run the tools using the following commands:

```streamlit run data_ingestion_app.py``` to run the data ingestion and processing portal

```streamlit run app.py``` to run the QA tool

```streamlit run maps_app.py``` to run the maps tool

#### ```/data``` contains all the externald data sources used in the maps tool

#### ```/data_ingestion_helpers``` contains the helper functions used in the data ingestion process. Each run of the data ingestion process will do the following:

1. Save the new Climate Action Plan to the CAPS folder
2. Collects the metatdata of the city (City, State, County, and City Center Coordinates) and updates the city_county_mapping.csv file
3. Generates a summary of the Climate Action Plan and stores it in the CAPS_Summaries folder
4. Creates the vector stores of the Climate Action Plan used in the QA tool (Individual, Summary and Combined Vector Stores)
5. Queries an LLM to update the climate actions plans dataset in climate_actions_plans.csv
6. Updates the CAPS plans list in caps_plans.csv
7. Re-runs the maps_data.py script to update the data powering the maps tool

#### ```/batch_scripts``` contains scripts that can be run to batch process CAPs. 

```batch_summary_generation.py``` generates summaries for all CAPs in the CAPS folder and saves them in the CAPS_Summaries folder

```caps_directory_reader.py``` reads in the CAPS plans in the CAPS folder and saves the data to a csv file called caps_plans.csv

```census_county_data.py``` reads in the census data and saves the data to a csv file called us_counties.csv which is used by the data ingestion tool

```create_vector_stores.py``` creates the vector stores of the Climate Action Plan used in the QA tool (Individual, Summary and Combined Vector Stores)

```dataset_generation.py``` queries an LLM to create the climate actions plans dataset in climate_actions_plans.csv

In most cases, these batch process files will not need to be run. 

#### ```/maps_helpers``` contains the helper functions used in the maps tool and stores the data powering the maps tool

To run the tool, in a terminal run ```streamlit run app.py```. Please ensure that all necessary packages have been installed as per the ```requirements.txt``` file. Necessary packages can be installed using pip: ```pip install -r requirements.txt``` 

The ```Prompts``` folder contains all the system prompt templates used in the tool. These can be modified to modify the behavior of the tools. 

## Project logistics

**Sprint planning**: Every Monday at 10-10:30am on [Zoom](https://vanderbilt.zoom.us/j/98561891048?pwd=VRIXN9QgykKV4HhblNLSCqu6UwKS6Z.1&from=addon). 

**Backlog Grooming**: NA / as needed. 

**Sprint Restrospective**: Every Friday 1:30-2pm on [Zoom](https://vanderbilt.zoom.us/j/91271107413?pwd=CESI9izE4x3Mcshv2DqNyAa7nG0GUr.1&from=addon). 

**Demos**: Every Friday at 3pm on Zoom as well as in person at the DSI.  

**Data location**: [Climate Policy Data](https://vanderbilt365-my.sharepoint.com/:f:/g/personal/ethan_i_thorpe_vanderbilt_edu/Eu8eb1jCuJpKoSTcq--22E4BSKa8mQxXrjD8p-2wrlX_hQ?e=P0j56t)

**Slack channel**: climate-policy on Data Science TIP slack organization. Please check your email for an invite. 

## Resources 

Provide any useful resources to get readers up to speed with the project here. 

* **LangChain**: Please see [LangChain Tutorials](https://python.langchain.com/docs/tutorials/)
* **Python usage**: Whirlwind Tour of Python, Jake VanderPlas ([Book](https://learning.oreilly.com/library/view/a-whirlwind-tour/9781492037859/), [Notebooks](https://github.com/jakevdp/WhirlwindTourOfPython))
* **Data science packages in Python**: [Python Data Science Handbook, Jake VanderPlas](https://jakevdp.github.io/PythonDataScienceHandbook/) 
* **HuggingFace**: [Website](https://huggingface.co/transformers/index.html), [Course/Training](https://huggingface.co/course/chapter1), [Inference using pipelines](https://huggingface.co/transformers/task_summary.html), [Fine tuning models](https://huggingface.co/transformers/training.html)
* **fast.ai**: [Course](https://course.fast.ai/), [Quick start](https://docs.fast.ai/quick_start.html)
* **h2o**: [Resources, documentation, and API links](https://docs.h2o.ai/#h2o)
* **nbdev**: [Overview](https://nbdev.fast.ai/), [Tutorial](https://nbdev.fast.ai/tutorial.html)
* **Git tutorials**: [Simple Guide](https://rogerdudler.github.io/git-guide/), [Learn Git Branching](https://learngitbranching.js.org/?locale=en_US)
* **ACCRE how-to guides**: [DSI How-tos](https://github.com/vanderbilt-data-science/how-tos)  

## Contact Info

Project Lead: [Umang Chaudhry](mailto:umang.chaudhry@vanderbilt.edu), Senior Data Scientist, Vanderbilt Data Science Institute  
PI: [Dr. JB Ruhl](mailto:jb.ruhl@vanderbilt.edu), David Daniels Allen Distinguished Chair in Law, Vanderbilt University Law School  
Project Manager: [Isabella Urquia](mailto:isabella.m.urquia@vanderbilt.edu)  
Team Members: [Ethan Thorpe](mailto:ethan.i.thorpe@vanderbilt.edu), [Mariah Caballero](mailto:mariah.d.caballero@vanderbilt.edu), Harmony Wang, Xuanxuan Chen, Aparna Lakshmi
