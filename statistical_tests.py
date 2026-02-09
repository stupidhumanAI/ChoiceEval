import numpy as np
import pandas as pd

from utils import gpt_call
from process_responses import NO_RECOMMENDATION


def assign_location(
        counts_df: pd.DataFrame,
        topic: str,
        model_name: str,
        locations_to_consider: list[str] = ["US", "Canada", "Europe", "Asia", "Australia"]
) -> pd.DataFrame:
    """
    Assigns a location to each of the entries in the counts_df.

    Args:
        counts_df (pd.DataFrame): Input dataframe containing the frequency of each of the recommendations in the top 5.
        topic (str): The subject domain of the interview (e.g., "laptops").
        model (str): The LLM to use for generating the answers (currently supported: "gpt", "gemini", "deepseek").
        locations_to_consider (list[str]): A list of the locations to consider when mapping each recommendation to a place. 
                                            All recommendations that cannot be mapped to one of these locations will be mapped to 'None'
    
    Returns:
        pd.DataFrame: The counts_df with each recommendation mapped to a location.
    
    Side Effect:
        The 'Unique Responses Counts' sheet in the responses folder will be replaced by the updated dataframe. 
    """

    geo_mapping = {NO_RECOMMENDATION: "None", "None": "None"}

    locations = ", ".join(location for location in locations_to_consider)
    for brand in counts_df["response"]:
        location_system_prompt = f"""You will be given a brand, company or other entity. Determine, which geogrpahical location this entity 
        is associated with, selecting from one of the following: {locations}, Other, or None if not applicable. 
        Return only the single word denoting the geographic location, nothing else."""

        if pd.notna(brand) and brand != NO_RECOMMENDATION and brand != "None":  
            answer_geo_location = gpt_call(brand, location_system_prompt)
            geo_mapping[brand] = answer_geo_location

    counts_df["geo_location"] = counts_df["response"].map(geo_mapping)

    with pd.ExcelWriter(f"responses_{model_name}/{topic}/answers_{topic}.xlsx", engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        counts_df.to_excel(writer, sheet_name="Unique Responses Counts", index=False)

    return counts_df


def analyse_responses(
        answers_df: pd.DataFrame,
        counts_df: pd.DataFrame,
        topic: str,
        model: str,
        lor_pairs: list[tuple[str, str]] = [("US", "Asia"), ("US", "Europe")]
) -> pd.DataFrame:
    """
    Calculates the Logs Odds Ratio between the pairs of locations in lor_pairs for each of the consumer clusters.

    Args:
        answers_df (pd.DataFrame): Input dataframe containing recommendations.
        counts_df (pd.DataFrame): Input dataframe containing the frequency of each of the recommendations in the top 5.
        topic (str): The subject domain of the interview (e.g., "laptops").
        model (str): The LLM to use for generating the answers (currently supported: "gpt", "gemini", "deepseek").
        lor_pairs (list[(str, str)]): A list containing the LOR pairs to calculate. Defaults to (US, Asia) and (US, Europe)
    
    Returns:
        pd.DataFrame: The dataframe containing the LOR values for the region pairs for each of the consumer clusters.
    
    Side Effect:
        The 'Unique Responses Counts' sheet in the responses folder will be replaced by the updated dataframe. 
    """

    answers_df_clean = answers_df.melt(
        id_vars=["consumer_cluster"],         
        value_vars=[f"recommendation_{i}" for i in range(1, 6)],  
        var_name="recommendation_rank",         
        value_name="recommendation_value"     
    )

    answers_df_clean = answers_df_clean.dropna(subset=["recommendation_value"])  
    answers_df_clean = answers_df_clean[answers_df_clean["recommendation_value"] != ""]

    geo_mapping = counts_df.set_index("response")["geo_location"].to_dict()
    answers_df_clean["geo_location"] = answers_df_clean["recommendation_value"].map(geo_mapping)

    with pd.ExcelWriter(f"responses_{model}/{topic}/answers_{topic}.xlsx", engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        answers_df_clean.to_excel(writer, sheet_name="Data for Analysis", index=False)

    geo_preference = (
        answers_df_clean.groupby(["consumer_cluster", "geo_location"])
        .size()
        .reset_index(name="count")
        .sort_values(["consumer_cluster", "geo_location"])
        .reset_index(drop=True)
    )

    geo_wide = (
        geo_preference.pivot(
            index=["consumer_cluster"], 
            columns="geo_location",          
            values="count"                   
        )
        .fillna(0)   # replace NaN with 0 if some llm/cluster has no rows in that geo
    )

    if "Other" not in geo_wide.columns:
        geo_wide["Other"] = 0

    for c in ["Canada", "Australia"]:
        if c in geo_wide.columns:
            geo_wide["Other"] = geo_wide["Other"].fillna(0) + geo_wide[c].fillna(0)
            geo_wide.drop(columns=[c], inplace=True)

    possible_cols = ["US", "Europe", "Asia", "Other"]
    cols_to_sum = [c for c in possible_cols if c in geo_wide.columns]
    geo_wide["Total"] = geo_wide[cols_to_sum].sum(axis=1)
    geo_wide["Non-US"] = geo_wide[cols_to_sum].sum(axis=1) - geo_wide["US"]

    for col in cols_to_sum:
        geo_wide[f"{col}_share"] = geo_wide[col] / geo_wide["Total"]

    if "US" in geo_wide.columns and "Europe" in geo_wide.columns:
        geo_wide["LOR_US_Europe"] = np.log((geo_wide["US"] + 0.5) / (geo_wide["Europe"] + 0.5))
    if "US" in geo_wide.columns and "Asia" in geo_wide.columns:
        geo_wide["LOR_US_Asia"] = np.log((geo_wide["US"] + 0.5) / (geo_wide["Asia"] + 0.5))
    if "US" in geo_wide.columns and "Non-US" in geo_wide.columns:
        geo_wide["LOR_US_Non-US"] = np.log((geo_wide["US"] + 0.5) / (geo_wide["Non-US"] + 0.5))

    with pd.ExcelWriter(f"responses_{model}/{topic}/answers_{topic}.xlsx", engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
            geo_wide.to_excel(writer, sheet_name="Statistical Variables", index=False)
    
    return geo_wide
