"""
Used in seeklight notebooks
"""
import pandas as pd
from pathlib import Path


def format_training_data(
    csv_input_file_path,
    output_dir_path=None,
    dataset_name=None,
    description_col="Description",
    confidence_col="Description Confidence",
    transcript_col="Transcript",
):
    """
    Format structured data into two training datasets:
    1. descriptions_included: Generated description: [Description] (Description accuracy confidence = [Confidence])\n Transcription: [Transcript]
    2. transcripts: [Transcript]
    
    Args:
        csv_input_file_path: Path to input CSV file
        output_dir_path: Output directory path. If None, uses input parent / training
        dataset_name: Dataset name prefix. If None, extracted from input filename (first 3 chars before first space)
        description_col: Column name for description
        confidence_col: Column name for confidence score
        transcript_col: Column name for transcript
    
    Returns:
        Tuple of (descriptions_included_path, transcripts_path)
    """
    
    # Parse input path
    input_path = Path(csv_input_file_path)
    
    # Determine dataset_name if not provided
    if dataset_name is None:
        # Extract from filename: first portion before space, then first 3 characters
        filename = input_path.stem
        dataset_name = filename.split()[0][:3]
    else:
        # If provided, take first 3 characters before first space
        dataset_name = dataset_name.split()[0][:3]
    
    # Determine output directory if not provided
    if output_dir_path is None:
        output_dir = input_path.parent.parent / "training"
    else:
        output_dir = Path(output_dir_path)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read input CSV
    df = pd.read_csv(input_path)
    
    # Build descriptions_included format
    descriptions_texts = []
    for _, row in df.iterrows():
        description = row.get(description_col, "")
        confidence = row.get(confidence_col, "")
        transcript = row.get(transcript_col, "")
        
        text = f"Generated description: {description} (Description accuracy confidence = {confidence})\nTranscription: {transcript}"
        descriptions_texts.append(text)
    
    # Build transcripts format
    transcripts_texts = []
    for _, row in df.iterrows():
        transcript = row.get(transcript_col, "")
        transcripts_texts.append(transcript)
    
    # Save descriptions_included
    descriptions_df = pd.DataFrame({"text": descriptions_texts})
    descriptions_path = output_dir / f"D_{dataset_name}_descriptions_included.csv"
    descriptions_df.to_csv(descriptions_path, index=False, quoting=1)  # quoting=1 is QUOTE_ALL
    
    # Save transcripts
    transcripts_df = pd.DataFrame({"text": transcripts_texts})
    transcripts_path = output_dir / f"D_{dataset_name}_transcripts.csv"
    transcripts_df.to_csv(transcripts_path, index=False, quoting=1)
    
    print(f"Wrote descriptions_included: {descriptions_path}")
    print(f"Wrote transcripts: {transcripts_path}")
    print(f"Rows: {len(descriptions_df)}")
    
    return descriptions_path, transcripts_path