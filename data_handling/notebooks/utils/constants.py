"""
Constants for file paths
"""

class Paths:
    # Path locations for data access via notebooks
    mccray_folder = r"../../data/mccray/"

    mccray_original_metadata = mccray_folder+ r"original_data/McCray metadata.xlsx"
    mccray_modified_metadata = mccray_folder + r"changed_data/McCray+.xlsx"

    mccray_with_trscp = mccray_folder + r"changed_data/trscp_subsets/McCray (with transcripts).xlsx"

    mccray_1940s_subset = mccray_folder + r"changed_data/decade_subsets/McCray (1940s).xlsx"
