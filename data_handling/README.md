# Data handling of USC digital collections metadata

## Notebooks setup 
### Setup venv & notebook kernel
#### Create a virtual environment
In the command line for our project directory, (../AspireAI_LLM_Project/), create a python virtual enviroment (purpose of a vir env is to isolate project libs/packages all into one place)
```bash 
python -m venv venv
```
#### Activate venv and install required libaries
```bash 
venv/scripts/activate
pip install -r requirements.txt
```
#### Make venv useable as notebooks kernel (via ipykernel)
'ipykernel' was installed as one of the packages in the venv, now we can use that to allow our packages to be useable in notebooks (as a kernel)
```bash
python -m ipykernel install --user --name=venv_aspire --display-name "Python (venv_aspire)"
```
Then exit and relaunch VS Code (or other notebook supporting IDE)
#### Notebooks are setup
* Choose 'Python (venv_aspire)' as the kernel when using any notebook for this project 
* Add new libaries to the requirements.txt
* Activate the venv and install the requirements.txt anytime crirtical libraries are added to the project 

### Notebook workflow
###### *Note:* Rounded rectanges = excel file, regular rectangles = notebook, circle = missing step in proccess 
```mermaid
flowchart TD
    A([Start: Raw Metadata]) --> B[Add **fields**]

    B --> C[Locate **Messy** Data; adding additional fields and creating visulizations on overall data standing]
    C --> D[**Split** the data based upon messiness]

    D --> Y([**Cleaner Transcripts**; no unusual patterns and substaintial in length])

    Y --> G(**Final Cleaning**: Standardize formating & apply context aware spell checks; still need to test better spell checkers, this is currently a manual reviewal)

    D --> F([**Messiest/Undesireable Transcripts**; unusual patterns or short in length])
    F --> H(**Intermediary Cleaning**: Pattern removal; can also add subsutuion of common patterns)
    H --> L(**Filtering**: Remove undiserables -- the shortest transcripts and ones with little to no common English words)
    L --> M([**Undesirable** Transcripts])
    M --> O((need implement pipeline to determine if documents are truly blank or if they need rescanned))



    L --> I([**Semi-clean** Transcripts])
    I --> J((... **?** need ways to get up to par with the 'cleaner' transcripts))
    J --> G

    G --> K([**Absolute Clean** Transcripts])



    
```

## util folder
### Importing modules
