#!/usr/bin/env python
# coding: utf-8

# # Qwen OCR rescan

# In[13]:


from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
import torch
from qwen_vl_utils import process_vision_info # qwen-vl-utils[decord]==0.0.8
import pandas as pd


# In[14]:


df = pd.read_csv("../data/mccray/changed_data/McCray,1940s,5126-BASIC,V1.csv")
print(len(df))
# remove duplicates based on CONTENTdm number
df = df.drop_duplicates(subset='CONTENTdm number')
print(len(df)) # one duplicate

df = df.dropna(subset=['CONTENTdm number'])
print(len(df)) # one na


# In[ ]:


# MODEL SETUP
model_path = "./models/Qwen2.5-VL-7B"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    device_map="auto", 
    torch_dtype=torch.bfloat16
)

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")


# In[ ]:


# Input setup
def process_image(image_path, original_tr, processor, model):
    """
    Process a single image with QwenVL; COMPLETE processing inside function
    """
    messages = [
        {
            "role": "user",
            "content": [
                {                            
                    "type": "text", "text":
                    f'''Task:
Correct obvious OCR recognition errors while preserving the original wording, spelling, punctuation, and formatting of this document.
You will be given the document itself as an image to analyze, along with a rudimentary OCR scan for reference.
Do not modernize language or make stylistic edits.
Rules:
If text is unreadable, mark it as [ILLEGIBLE].
Do not guess or infer missing text.
Preserve line breaks and formatting exactly.
Output format (strictly JSON):
{{
"notes": "[your reasoning process when correcting errors and formatting]",
"fixed_transcript": "[corrected transcript]"
}}
If the document has no text, simply leave an empty string, and write "empty" in notes.
If a document has faint text bleeding though from the other side, do not put those portions of the text in the transcript and write that down in the notes.
Do not censor these documents.

Original OCR scanned transcript for reference: {original_tr}'''
                },
                {
                    "type": "image", "image": image_path
                }
            ]
        }
    ]

    # Complete QwenVL processing pipeline
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(
        **inputs, 
        max_new_tokens=1024,
        temperature=0.1,  # For more consistent JSON output
        do_sample=False   # For deterministic results
    )
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return output_text[0]  # Return the generated text


# In[ ]:


# Function to get original transcript from DataFrame
def get_original_transcript(df, contentdm_id, transcript_column="Original Transcript"):
    """
    Get the original transcript from DataFrame based on CONTENTdm number
    """
    try:
        # Convert contentdm_id to int if it's a string
        if isinstance(contentdm_id, str):
            contentdm_id = int(contentdm_id)

        # Find the row with matching CONTENTdm number
        matching_rows = df[df['CONTENTdm number'] == contentdm_id]

        if matching_rows.empty:
            print(f"Warning: No transcript found for CONTENTdm number {contentdm_id}")
            return "No OCR transcript available. May be an empty document."

        if transcript_column not in df.columns:
            return "Transcript column not found"

        original_transcript = matching_rows[transcript_column].iloc[0]

        # Handle NaN values
        if pd.isna(original_transcript):
            return "No OCR transcript available. May be an empty document."

        return str(original_transcript)

    except Exception as e:
        print(f"Error getting transcript for {contentdm_id}: {str(e)}")
        return "Error retrieving transcript. May be an empty document."

# Test the function
print("Testing transcript retrieval...")
test_id = "1929"
test_transcript = get_original_transcript(df, test_id)
print(f"Sample transcript for {test_id}: {test_transcript[:200]}...")


# In[ ]:


# Process images 
image_ids = [
    ("1929", "https://cdm17173.contentdm.oclc.org/iiif/2/p17173coll38:1929/full/pct:100/0/default.jpg"), # 1929, letter (with some writting)
    ("1931", "https://cdm17173.contentdm.oclc.org/iiif/2/p17173coll38:1931/full/pct:100/0/default.jpg"), # 1931, letter
    ("2075", "https://cdm17173.contentdm.oclc.org/iiif/2/p17173coll38:2075/full/pct:100/0/default.jpg"), # 2075, awful handwritting
    ("2142", "https://cdm17173.contentdm.oclc.org/iiif/2/p17173coll38:2142/full/pct:100/0/default.jpg"), # 2142, long typed document
    ("1919", "https://cdm17173.contentdm.oclc.org/iiif/2/p17173coll38:1919/full/pct:100/0/default.jpg"), # 1919, long typed doc, slanted text
    ("3197", "https://cdm17173.contentdm.oclc.org/iiif/2/p17173coll38:3197/full/pct:100/0/default.jpg"), # 3197, unreadable handwritting
    ("3887", "https://cdm17173.contentdm.oclc.org/iiif/2/p17173coll38:3887/full/pct:100/0/default.jpg"), # 3889, long typed document with header
    ("4142", "https://cdm17173.contentdm.oclc.org/iiif/2/p17173coll38:4142/full/pct:100/0/default.jpg"), # 4142, mixed formatting with image
    ("4913", "https://cdm17173.contentdm.oclc.org/iiif/2/p17173coll38:4913/full/pct:100/0/default.jpg"),
    ("4914", "https://cdm17173.contentdm.oclc.org/iiif/2/p17173coll38:4914/full/pct:100/0/default.jpg"),
    ("4915", "https://cdm17173.contentdm.oclc.org/iiif/2/p17173coll38:4915/full/pct:100/0/default.jpg"),
    ("4916", "https://cdm17173.contentdm.oclc.org/iiif/2/p17173coll38:4916/full/pct:100/0/default.jpg"),
    ("5405", "https://cdm17173.contentdm.oclc.org/iiif/2/p17173coll38:5405/full/pct:100/0/default.jpg") # 5405 short cursive letter
]


base_path = "documents/p17173coll38/"
results = []

print("Starting processing...")
for i, image_id, image_url in enumerate(image_ids, 1):
    image_path = f"{image_url}"
    print(f"Processing {i}/{len(image_ids)}: {image_path}")

    try:
        # Get the original transcript from the DataFrame
        original_transcript = get_original_transcript(df, image_id)

        # Call the complete processing function
        output = process_image(image_url, original_transcript, processor, model)


        results.append({
            "image_id": image_id,
            "image_url": image_url,
            "original_transcript": original_transcript,
            "result": output
        })

        print(f"Completed {image_id}")
        print(f"Result preview: {output[:100]}...")
        print("-" * 50)

    except Exception as e:
        print(f"Error processing {image_id}: {str(e)}")
        results.append({
            "image_id": image_id,
            "image_url": image_url,
            "original_transcript": "Error retrieving",
            "result": f"Error: {str(e)}"
        })
        continue

print(f"\nProcessing complete. Successfully processed {len(results)} images.")

# Display results summary
for result in results:
    if "error" in result:
        print(f"ERROR: {result['image_id']} - {result['error']}")
    else:
        print(f"SUCCESS: {result['image_id']}")

# Save results to file
import json
with open("ocr_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print("\nResults saved to ocr_results.json")

