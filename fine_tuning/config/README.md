# Config Files

  

Config files are separated into 3 types:
1. Fine-tuning in `ft/`
2. Inference in `inference/` # Maybe remove
3. Evaluation in `eval/`
4. Data in `data/`

## Models
`models/<identifier>.json`

### Format
```json
{
    "path": "",     // Path is relative to `base_dir` defined in the .env file
    "type": "",     // instruct | base
    "format": "",   // meta | google | openai // Used to determine prompt formats for instruct models
}
```

## Fine-tuning
`ft/<type>.<version>.json`

### Types:

| Name                 | Type Identifier | Description                                                               |
| -------------------- | --------------- | ------------------------------------------------------------------------- |
| LoRa Self-Supervised | `lora_ss`       | LoRa fine-tuning on raw text (no input/output pairs, unlabeled text data) |
| LoRa Supervised      | `lora_s`        |                                                                           |
| ...                  | ...             | ...                                                                       |
### Versioning:

New versions should be created when we want to test multiple parameters for fine-tuning scripts. Versioning starts with `0`.

## Evaluation



## Results

Results will be saved in 


