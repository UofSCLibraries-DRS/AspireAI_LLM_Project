from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
import torch
from qwen_vl_utils import process_vision_info

model_path = "../model/Qwen/Qwen2.5-VL-7B-Instruct"

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)

# default processer
processor = AutoProcessor.from_pretrained(model_path)

# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

messages = [
        {
            "role": "user",
            "content": [
                {                            
                    "type": "text", "text":
                    '''Task:
Correct obvious OCR recognition errors while preserving the original wording, spelling, punctuation, and formatting of this document.
You will be given the document itself as an image to analyze, along with a rudimentary OCR scan for reference.
Do not modernize language or make stylistic edits.
Rules:
If text is unreadable, mark it as [ILLEGIBLE].
Do not guess or infer missing text.
Preserve line breaks and formatting exactly.
Output format (strictly JSON):
{
"notes": "[your reasoning process when correcting errors and formatting]",
"fixed_transcript": "[corrected transcript]"
}
If the document has no text, simply leave an empty string, and write "empty" in notes.
If a document has faint text bleeding though from the other side, do not put those portions of the text in the transcript and write that down in the notes.
Do not censor these documents.

Original OCR scanned transcript for reference: 
Progressive Democratic Party    Executive Officers    JOHN H. MCCRAY  STATE CHAIRMAN  COLUMBIA. S. C.    REV. W. L. WILSON  1ST VICE-CHAIRMAN  SPARTANBURG, S. C.    DR. R. W. SPARKS  2ND VICE-CHAIRMAN  DARLINGTON, S. C.    JOHN H. GREEN  3RD VICE-CHAIRMAN  CHARLESTON, S. C.    REV. JAS. J. ABNEY  4TH VICE-CHAIRMAN  AIKEN, S. C.    MRS. A. B. WESTON  SECRETARY  COLUMBIA, S. C.    REV. E. E. GAULDEN  1ST ASSISTANT SECRETARY  NEWBERRY, S. C.    JAMES PRIOLEAU  2ND ASSISTANT SECRETARY  Georgetown, s. c.    J. C. ARTEMUS  TREASURER  COLUMBIA, S. C.    OSCEOLA E. McKAINE  EXECUTIVE SECRETARY  SUMTER. S. C.    lO22'/i Washington Street    COLUMBIA 20, S. C.    May 3,1945    TO THE STATE COMMITTEE;    We are urging each member of the State Committee  to attend in person or, by proxy, a special meeting in  the Masonic Temple, 1125 Washington Street, Columbia,  at noon on Friday, May 11,1945.    Because at this meeting we shall, attempt tp formu-  late definite policies to which the Progressive Demo┬¼  cratic Party shall be bound and, consider certain other  matters now of first importance, we are trusting that  each county will be represented;under the Constitution,  each county and district chairman or some other named  person from the county, and each member among executive  officers are members of the State Committee, We are  asking inaddition that all members of the Speakers*  Bureau be present.    Among matters to be disposed ┬⌐f are: 1. Policies-  few of us know what or what not to do in the name of the  party, 2. Permanent quarters -development of permanent  quarters and a full time worker as ordered by the State  Committee in its January 1945 meeting. 3. Birthday -On  May 24th it is proposed that appropriate celebrations, in  parties, dinners, etc. be held by wards and clubs at the  end on one yearΓÇÖs existence. 4, Enrolment - few of us  know who are our members, 5. Registration - what shall  we do about new registration problems? 6. Certificates -  it is proposed that we issue charters to county, club  and ward organizations and, membership cards. 7. Temporary  office ΓÇöas a means of facilitating the permanent office,  staffed with onq full time secretary-clerk.    These are vital matters and should command concern  from every party member. We urge you to be present or, to  be represented. Please notify the state chairman immed┬¼  iately whether you or another person will represent you.    ┬╗ ' Sincerely yours,    Jo'-hn- H, -McCray^    - ' ' STATE CHAIRMAN    Osceola E, McKaine/^X  EXECUTIVE SECRETARY A'''
                },
                {
                    "type": "image", "image": "https://cdm17173.contentdm.oclc.org/iiif/2/p17173coll38:1929/full/pct:100/0/default.jpg"
                }
            ]
        }
    ]

# Preparation for inference
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
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)