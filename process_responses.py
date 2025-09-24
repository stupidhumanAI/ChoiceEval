import ast
import json
from pathlib import Path
import re
from typing import Callable
import numpy as np
import pandas as pd
from pydantic import BaseModel
from openpyxl.utils.exceptions import IllegalCharacterError
from utils import deepseek_call, gpt_call, gemini_call


class Recommendations(BaseModel):
    recommendation_1: str
    recommendation_2: str
    recommendation_3: str
    recommendation_4: str
    recommendation_5: str

def _list_entities(text: str, topic: str) -> dict[str, str]:
    """
    Extract up to five {topic} brands or providers from a text, in order of appearance.

    Args:
        text (str): The source text to analyze.
        topic (str): The domain label that contextualizes the entities (e.g., "laptops",
            "insurance providers").

    Returns:
        dict[str, str]: A dictionary with keys "recommendation_1" through
        "recommendation_5" mapped to the extracted brand/provider names (or
        "No recommendation" where applicable).
    """

    system_prompt_entities = f"""You will be given a text about {topic}.
        Act as 5 independent experts. Analyze the text and list the first 5 {topic} brands or providers
        recommended in the text. If a specific product or service is mentioned, list the brand or provider, not the product 
        or service itself.
        Extremely Important: List the {topic} brands or providers in the exact order they are mentioned in the text.
        Provide the output a simple majority agrees upon.

        Provide the output in a dictionary with the following format: """ + """
        {
         "recommendation_1": recommendation 1,
         "recommendation_2": recommendation 2,
         "recommendation_3": recommendation 3,
         "recommendation_4": recommendation 4,
         "recommendation_5": recommendation 5,
        }
        """ + f"""
        If less then 5 {topic} brands or providers are mentioned, return 'No recommendation' for the missing recommendations.
        If no {topic} brands or providers are mentioned, return 'No recommendation' in place of all 5 recommendations.
        """
    entities_dictionary = gpt_call(text, system_prompt_entities, output_text_format = Recommendations)
    print(entities_dictionary)
    return ast.literal_eval(entities_dictionary)


def _sanitize_dataframe(df: pd.DataFrame, truncate: bool = True) -> pd.DataFrame:
    """Apply sanitizer to all string cells in the dataframe."""
    _ILLEGAL_XML_CHARS_RE = re.compile(r'[\x00-\x08\x0B\x0C\x0E-\x1F]')
    return df.applymap(lambda v: _ILLEGAL_XML_CHARS_RE.sub('', v) if isinstance(v, str) else v)


def create_interview_answers(topic: str, model: str, questions: dict[str, list[str]]) -> pd.DataFrame:
    """
    Generate a Q&A dataset by asking each question to an LLMs (currently: GPT, Gemini, DeepSeek),
    capturing answers, and extracting up to five ordered recommendations per answer.

    For every input question, the function:
      1) Calls the specified model.
      2) Stores the raw answer, the LLM identifier ("gpt" or "gemini"), and the topic.
      3) Extracts entities via `list_entities(answer, topic)` and fills `recommendation_1` … `recommendation_5`.

    The output is a tidy DataFrame with one row per question, containing: 
    consumer cluster, question, topic, llm, answer, and five recommendation columns. 
    The DataFrame is also saved to an Excel file at `responses/{topic}/answers_{topic}.xlsx` (sheet: "Raw Data").

    Args:
        topic (str): The subject domain of the interview (e.g., "laptops").
        questions (dict[str, list[str]]): A mapping of consumer-cluster names to the list of questions to ask for that cluster.
        model (str): The LLM to use for generating answers ("gpt", "gemini", "deepseek").

    Returns:
        pandas.DataFrame: A DataFrame with columns:
            - consumer_cluster (str)
            - question (str)
            - topic (str)
            - llm (str: "gpt", "gemini" or "deepseek")
            - answer (str)
            - recommendation_1 … recommendation_5 (str)

    
    """
    model_name = (model or "").strip().lower()
    caller: Callable[[str], str]

    if model_name == "gpt":
        caller = gpt_call
    elif model_name == "gemini":
        caller = gemini_call
    elif model_name == "deepseek":
        caller = deepseek_call
    else:
        raise ValueError(
            f"Unknown model_name '{model}'. "
            "Supported: 'gpt', 'gemini', 'deepseek'."
        )

    out_dir = Path(f"responses/{topic}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"answers_{topic}.xlsx"
    if out_file.exists():
        answers_df = pd.read_excel(out_file, sheet_name="Raw Data")
    else:
        answers_df = pd.DataFrame([(cluster, q) for cluster, qs in questions.items() for q in qs], columns=['consumer_cluster', 'question'])
        answers_df["topic"] = topic
        answers_df["llm"] = model_name
        answers_df["answer"] = None
        for i in range(1, 6):
            answers_df[f"recommendation_{i}"] = None

    for index, row in answers_df.iterrows():
        existing = row.get("answer")
        if pd.notna(existing) and str(existing).strip() != "":
            continue  # skip already processed rows

        interview_question = row['question']
        answer = caller(interview_question)

        entities = _list_entities(answer, topic)
        answers_df.at[index, 'answer'] = answer
        answers_df.loc[index, entities.keys()] = pd.Series(entities)

        try:
            answers_df.to_excel(f"responses/{topic}/answers_{topic}.xlsx", sheet_name="Raw Data", index=False)
        except IllegalCharacterError as e:
            print(f"IllegalCharacterError while writing Excel. Detecting & sanitizing... Error: {e}")
            answers_df = _sanitize_dataframe(answers_df, truncate=True)
            answers_df.to_excel(f"responses/{topic}/answers_{topic}.xlsx", sheet_name="Raw Data", index=False)

    return answers_df


class NormalisationDictionary(BaseModel):
    duplicated_value: str
    original_value: str 

def normalise_responses(answers_df: pd.DataFrame, topic: str) -> None:
    """
    Detect and consolidate duplicate brand/provider names across recommendation columns. When the model flags a duplicate, the longer string
    (`duplicated_value`) is mapped to the shorter canonical form (`original_value`). The mapping is applied to the recommendations and saved to disk.

    Args:
        answers_df (pd.DataFrame): Input dataframe containing recommendations.
        topic (str): The subject domain of the interview (e.g., "laptops").

    Side Effect:
        - Writes `normalisation.json` into reponses folder.
    """

    rec_cols = [col for col in answers_df.columns if col.startswith("recommendation_")]
    all_recs = answers_df[rec_cols].values.ravel()     
    counts = pd.Series(all_recs).value_counts()
    counts_df = counts.reset_index()
    counts_df.columns = ["value", "count"]

    system_prompt_normalisation = f"""You will receive 2 {topic} brands or providers. 

    If both values refer to the same brand or provider:
    - Let [duplicated_value] be the longer value.
    - Let [original_value] be the shorter value. 

    Otherwise let [duplicated_value] = [original_value] = "None". 
    If either of the values is 'No recommendation' let [duplicated_value] = [original_value] = 'None.' """ + """

    Output strictly a valid json in this structure:
    {
        "duplicated_value": [duplicated_value],
        "original_value": [original_value]
    }
    """

    normalisation_mapping = {}
    normalised_counts_df = counts_df.copy()

    def apply_mapping_if_valid(d: dict):
        dv = d.get("duplicated_value")
        ov = d.get("original_value")
        if dv and ov and dv != ov and dv != 'No recommendation':
            normalisation_mapping[dv] = ov
            normalised_counts_df.replace({dv: ov}, inplace=True)
            answers_df.replace({dv: ov}, inplace=True)

    if len(counts_df) >= 2:
        user_prompt_normalisation = counts_df.iloc[0]["value"] + ", " + counts_df.iloc[1]["value"]
        mapping_dictionary = ast.literal_eval(gpt_call(user_prompt=user_prompt_normalisation, system_prompt=system_prompt_normalisation, output_text_format=NormalisationDictionary))
        apply_mapping_if_valid(mapping_dictionary)

    for i in range(2, len(counts_df)):
        vi = counts_df.iloc[i]["value"]
        # skip if this value already got normalized by previous mappings
        if vi in normalisation_mapping:
            continue
        if (pd.isna(vi) or vi.strip() == "" or vi == "No recommendation"):
            continue
        for j in range(0, min(i,15)):
            vj = normalised_counts_df.iloc[j]["value"]
            user_prompt_normalisation = vi + ", " + vj
            try:
                mapping_dictionary = ast.literal_eval(gpt_call(user_prompt=user_prompt_normalisation, system_prompt=system_prompt_normalisation, output_text_format=NormalisationDictionary))
                apply_mapping_if_valid(mapping_dictionary)
                normalised_counts_df = normalised_counts_df.groupby('value', as_index=False, sort=False).agg(count=('count', 'sum')).sort_values(by="count", ascending=False).reset_index(drop=True)
                if vi in normalisation_mapping:
                    break
            except Exception as e:
                print(f"Error processing vi='{vi}', vj='{vj}': {e}")
                continue

    normalisation_json_path = f"responses/{topic}/normalisation.json"
    with open(normalisation_json_path, "w") as f:
        json.dump(normalisation_mapping, f, indent=4)
    
    return normalisation_mapping


def apply_normalisation(answers_df: pd.DataFrame, topic: str, normalisation_mapping: dict) -> str:
    """
    Apply a given normalisation mapping to the recommendation columns of the answers dataframe.

    Args:
        answers_df (pd.DataFrame): Input dataframe containing recommendations.
        topic (str): The subject domain of the interview (e.g., "laptops").
        normalisation_mapping (dict): A mapping of duplicated brand/provider names to their canonical forms.

    Returns:
        pd.DataFrame: The dataframe with normalized recommendation columns.
    """
    rec_cols = [col for col in answers_df.columns if col.startswith("recommendation_")]
    for col in rec_cols:
        answers_df[col] = answers_df[col].map(normalisation_mapping).fillna(answers_df[col])

    melted = answers_df[rec_cols].melt(value_name="response", value_vars=rec_cols)
    responses_series = melted["response"]
    counts_df = responses_series.value_counts().rename_axis("response").reset_index(name="count")

    with pd.ExcelWriter(f"responses/{topic}/answers_{topic}.xlsx", engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        answers_df.to_excel(writer, sheet_name="Normalised Data", index=False)
        counts_df.to_excel(writer, sheet_name="Unique Responses Counts", index=False)

    return answers_df, counts_df


def assign_location(counts_df: pd.DataFrame, topic: str) -> str:
    geo_mapping = {"No recommendation": "None", "None": "None"}

    for brand in counts_df["response"]:
        location_system_prompt = """You will be given a brand, company or other entity. Your task is to determine, which geogrpahical location is this entity 
        assosiacted with. Select one of the following: US, Canada, Europe, Asia, Australia, Other or None if not applicable. 
        Return only a single word denoting the geo location, nothing else. 
        """
        if pd.notna(brand) and brand != "No recommendation" and brand != "None":  
            answer_geo_location = gpt_call(brand, location_system_prompt)
            geo_mapping[brand] = answer_geo_location

    counts_df["geo_location"] = counts_df["response"].map(geo_mapping)

    with pd.ExcelWriter(f"responses/{topic}/answers_{topic}.xlsx", engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        counts_df.to_excel(writer, sheet_name="Unique Responses Counts", index=False)

    return counts_df


def analyse_responses(answers_df: pd.DataFrame, counts_df: pd.DataFrame, topic: str) -> None:
    answers_df_clean = answers_df.melt(
        id_vars=["consumer_cluster"],         
        value_vars=[f"recommendation_{i}" for i in range(1, 6)],  
        var_name="recommendation_rank",         
        value_name="recommendation_value"     
    )

    threshold = 0.05 * 207
    brands_to_consider = counts_df[counts_df["count"] > threshold]["response"].to_list()

    answers_df_clean = answers_df_clean.dropna(subset=["recommendation_value"])  
    answers_df_clean = answers_df_clean[answers_df_clean["recommendation_value"] != ""]
    answers_df_clean = answers_df_clean[answers_df_clean["recommendation_value"].isin(brands_to_consider)]  

    geo_mapping = counts_df.set_index("response")["geo_location"].to_dict()
    answers_df_clean["geo_location"] = answers_df_clean["recommendation_value"].map(geo_mapping)

    with pd.ExcelWriter(f"responses/{topic}/answers_{topic}.xlsx", engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
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

    with pd.ExcelWriter(f"responses/{topic}/answers_{topic}.xlsx", engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
            geo_wide.to_excel(writer, sheet_name="Statistical Variables", index=False)