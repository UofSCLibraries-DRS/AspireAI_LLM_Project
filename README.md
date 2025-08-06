# AspireAI LLM Project

## Project Structure

```
AspireAI_LLM_Project/
├── data/mccray/
├── data_handling/
│   ├── utils/
│   │   └── contants.py
│   │   └── nb_metrics.py
│   │   └── ocr_cleaning.py
│   ├── notebooks/
│   │   ├── analytics/
│   │   ├── utils/
│   │   ├── 00-fields.ipynb
│   │   ├── 01-messy.ipynb
│   │   ├── 02-split.ipynb
│   │   ├── 03-decades.ipynb
│   │   ├── 04-cleaning.ipynb
│   │   └── mess-check.ipynb
│   └── README.md
├── llm_tuning/
│   └── README.md
├── .gitignore
├── requirements.txt
└── README.md
```

### Directories

- **`data/`** - All excel/csv files for project
- **`data_handling/`** - Data processing and analysis files; contains jupyter notebooks and python code to handle our metadata (manipulations, cleaning, etc.)
- **`llm_tuning/`** - Large language model fine-tuning components (currently just testing)

### Key Files

- **`requirements.txt`** - Project's Python package dependencies
- **Notebook Pipeline**:
  - `00-fields.ipynb` - Create additional metadata fields
  - `01-messy.ipynb` - Analyze messy characters in transcripts, adding columns to make distictions between roughness/cleaniness (useful for entire dataset and subsets)
  - `02-split.ipynb` - Split data by transcript length, resulting in 
  - `03-decades.ipynb` - Separate data by decades
  - `04-cleaning.ipynb` - Clean metadata (WIP)
  - `mess-check.ipynb` - Quick messy character checker for data subsets that already went through *01-messy.ipynb*
- **Python Utils** (helper classes/functions for notebooks)
  - `contants.py` 
  - `nb_metrics.py` - notebook metrics
  - `ocr_cleaning.py`