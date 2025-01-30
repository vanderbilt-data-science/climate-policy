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

To run the tool, in a terminal run ```streamlit run app.py```. Please ensure that all necessary packages have been installed as per the ```requirements.txt``` file. Necessary packages can be installed using pip: ```pip install -r requirements.txt``` 

The ```Prompts``` folder contains all the system prompt templates used in the tool. These can be modified to modify the behavior of the tools. 

## Project logistics

**Sprint planning**: Every Monday at 3-4pm on [Zoom](https://vanderbilt.zoom.us/j/99721970914?pwd=UK6icdNV4bKaXiVZAArk8aVatmzKEC.1&from=addon). 

**Backlog Grooming**: Every Wednesday at 10-11am on [Zoom](https://vanderbilt.zoom.us/j/98299561939?pwd=vl4Aa7HvmBoTCVR4QaaRlpRKevxYpo.1&from=addon). 

**Sprint Restrospective**: Every Friday 12:15-12:45pm on [Zoom](https://vanderbilt.zoom.us/j/97196543286?pwd=Km2IIUtF0fltijN5oQ92v9wtrNqKt4.1&from=addon). 

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
Team Members: [Ethan Thorpe](mailto:ethan.i.thorpe@vanderbilt.edu), [Mariah Caballero](mailto:mariah.d.caballero@vanderbilt.edu), Xuanxuan Chen, Aparna Lakshmi, Harmony Wang  
