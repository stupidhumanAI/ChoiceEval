import ast
import json
from pathlib import Path
import re
from typing import Callable
import pandas as pd
from pydantic import BaseModel
from openpyxl.utils.exceptions import IllegalCharacterError
from utils import deepseek_call, gpt_call, gemini_call

NO_RECOMMENDATION = "No recommendation"


class Recommendations(BaseModel):
    recommendation_1: str
    recommendation_2: str
    recommendation_3: str
    recommendation_4: str
    recommendation_5: str

def _list_entities(text: str, topic: str, normalisation_grouping: str) -> dict[str, str]:
    """
    Extract up to five {topic} brands or providers from a text, in order of appearance.

    Args:
        text (str): The source text to analyze.
        topic (str): The domain label that contextualises the entities (e.g., "laptops", "insurance providers").
        normalisation_grouping (sr): The high level grouping to be used for extracting responses (e.g country, city, brand, provider)

    Returns:
        dict[str, str]: A dictionary with keys "recommendation_1" through to "recommendation_5" mapped to the extracted topic names (or
        NO_RECOMMENDATION where applicable).
    """

    system_prompt_entities = f"""You will be given a text about {topic}.
    Act as 5 independent experts who each identify the {topic} {normalisation_grouping} mentioned in the text.
    Return a single consensus result (simple majority) listing the first five {normalisation_grouping} in the exact order they appear in the text.

    Hard rules (must follow exactly):
    1. List {normalisation_grouping} in the **exact order they are first mentioned** in the text.
    2. If a single mention contains multiple {normalisation_grouping} joined by connectors, treat them as separate {normalisation_grouping}. (e.g., "Air France and KLM" -> "Air France", "KLM").

    Return **only** one JSON object, nothing else, with exactly these keys and string values:
    {{
        "recommendation_1": "recommendation_1",
        "recommendation_2": "recommendation_2",
        "recommendation_3": "recommendation_3",
        "recommendation_4": "recommendation_4",
        "recommendation_5": "recommendation_5"
    }}

    If fewer than 5 distinct brands/providers are present, use the string "{NO_RECOMMENDATION}" for the missing slots.
    If no {topic} brands or providers are mentioned, return '{NO_RECOMMENDATION}' in place of all 5 recommendations."""

    return ast.literal_eval(gpt_call(text, system_prompt_entities, output_text_format = Recommendations))


def _sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply sanitizer to all string cells in the dataframe."""

    _ILLEGAL_XML_CHARS_RE = re.compile(r'[\x00-\x08\x0B\x0C\x0E-\x1F]')
    return df.applymap(lambda v: _ILLEGAL_XML_CHARS_RE.sub('', v) if isinstance(v, str) else v)


def create_interview_answers(topic: str, model: str, normalisation_grouping: str, questions: dict[str, list[str]]) -> pd.DataFrame:
    """
    Generates a Q&A dataset by sending each question to the model, and then extracting up to five ordered recommendations from the corresponding answer.

    Args:
        topic (str): The subject domain of the interview (e.g., "laptops").
        questions (dict[str, list[str]]): A mapping of consumer-cluster names to the list of questions to ask for that cluster.
        model (str): The LLM to use for generating the answers (currently supported: "gpt", "gemini", "deepseek").
        normalisation_grouping (sr): The high level grouping to be used for extracting responses e.g country, city, brand, provider

    Returns:
        pandas.DataFrame: A DataFrame with one row per question, containing: consumer cluster, question, llm answer, and five recommendation columns (recommendation_1 ... recommendation_5)

    Side Effect: 
        The DataFrame is also saved to an Excel file at `responses_{model}/{topic}/answers_{topic}.xlsx` (sheet: "Raw Data").
    """

    model_name = model.strip().lower()
    caller: Callable[[str], str]

    if model_name == "gpt":
        caller = gpt_call
    elif model_name == "gemini":
        caller = gemini_call
    elif model_name == "deepseek":
        caller = deepseek_call
    else:
        raise ValueError(f"Unknown model_name '{model}'. Models currently supported: 'gpt', 'gemini', 'deepseek'.")

    out_dir = Path(f"responses_{model_name}/{topic}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"answers_{topic}.xlsx"

    if out_file.exists():
        answers_df = pd.read_excel(out_file, sheet_name="Raw Data")
    else:
        answers_df = pd.DataFrame([(cluster, q) for cluster, qs in questions.items() for q in qs], columns=['consumer_cluster', 'question'])
        answers_df["answer"] = None
        for i in range(1, 6):
            answers_df[f"recommendation_{i}"] = None

    for i, row in answers_df.iterrows():
        existing = row.get("answer")
        if pd.notna(existing) and str(existing).strip() != "": # skip already processed rows
            continue

        interview_question = row['question']
        answer = caller(interview_question)

        entities = _list_entities(answer, topic, normalisation_grouping)
        answers_df.at[i, 'answer'] = answer
        answers_df.loc[i, entities.keys()] = pd.Series(entities)

        try:
            answers_df.to_excel(f"responses_{model_name}/{topic}/answers_{topic}.xlsx", sheet_name="Raw Data", index=False)
        except IllegalCharacterError as e:
            print(f"IllegalCharacterError while writing to Excel. Detecting & sanitizing the text. Error: {e}")
            answers_df = _sanitize_dataframe(answers_df)
            answers_df.to_excel(f"responses_{model_name}/{topic}/answers_{topic}.xlsx", sheet_name="Raw Data", index=False)

    return answers_df


def normalise_responses(answers_df: pd.DataFrame, topic: str, model: str, normalisation_grouping: str) -> dict:
    """
    Detect and consolidate duplicates across recommendation columns. 

    Args:
        answers_df (pd.DataFrame): Input dataframe containing recommendations.
        topic (str): The subject domain of the interview (e.g., "laptops").
        model (str): The LLM to use for generating the answers (currently supported: "gpt", "gemini", "deepseek").
        normalisation_grouping (str): The high level grouping to be used for normalisation e.g country, city, brand, provider

    Side Effect:
        Writes `normalisation.json` into the reponses folder.
    """

    rec_cols = [col for col in answers_df.columns if col.startswith("recommendation_")]
    all_recs = answers_df[rec_cols].values.ravel()     
    counts = pd.Series(all_recs).value_counts()
    counts_df = counts.reset_index()
    counts_df.columns = ["value", "count"]

    system_prompt_normalisation = f"""You are a data-normalisation engine for {topic}.
    You will receive a flat list of strings: brand names, provider names, countries, cities, attractions, product models, etc.
    Your job is to map each input string to a single canonical (simplest, highest-level) name according to the requested grouping: {normalisation_grouping}.

    Rules (apply in order):
    1. OUTPUT: Return **only** a single valid JSON object and nothing else. The JSON must map each original input exactly (as it appears) to its canonical value, e.g.:
    {{
        "Nike Air Max 90": "Nike",
        "paris": "France",
        "Eiffel Tower": "France"
    }}
    Do NOT include any mapping where the canonical value is exactly the same as the original value after normalisation. In other words: only output entries where the original should change.
    Also never create a mapping for "{NO_RECOMMENDATION}"
    2. Canonicalization logic by grouping:
    - If grouping is **country**: map cities, attractions, and country variants to the canonical English country name (e.g. "Paris" -> "France", "Eiffel Tower" -> "France", "UK" -> "United Kingdom").
    - If grouping is **city**: map attractions and neighbourhoods to their city (e.g. "Eiffel Tower" -> "Paris").
    - If grouping is **brand** or **provider**: map product models, sub-brands and variations to the parent brand/company (e.g. "Nike Air Max" -> "Nike", "Apple iPhone 12" -> "Apple").
    - If grouping is a different entity type, apply the same principle: map any more-specific term to the requested higher-level grouping.
    3. Normalisation details:
    - Prefer the common, short, human-readable canonical name (e.g., "France", "Nike", "San Francisco").
    - Remove trailing/leading whitespace when matching, ignore case when deciding canonical form, but preserve the original input as the JSON key exactly as given.
    - Resolve obvious synonyms/abbreviations and common misspellings if confident (e.g., "U.S.A." -> "United States").
    """
    gpt_response = gpt_call(user_prompt=str(counts_df["value"].unique()), system_prompt=system_prompt_normalisation)
    clean_response = re.sub(r"^```(?:json)?|```$", "", gpt_response.strip(), flags=re.MULTILINE).strip()
    clean_response = re.sub(r"```(?:json)?", "", clean_response).strip()
    normalisation_mapping = ast.literal_eval(clean_response)

    normalisation_json_path = f"responses_{model}/{topic}/normalisation.json"
    with open(normalisation_json_path, "w") as f:
        json.dump(normalisation_mapping, f, indent=4)
    
    return normalisation_mapping


def apply_normalisation(answers_df: pd.DataFrame, topic: str,  model: str, normalisation_mapping: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply a given normalisation mapping to the recommendation columns of the answers dataframe.

    Args:
        answers_df (pd.DataFrame): Input dataframe containing recommendations.
        topic (str): The subject domain of the interview (e.g., "laptops").
        model (str): The LLM to use for generating the answers (currently supported: "gpt", "gemini", "deepseek").
        normalisation_mapping (dict): A mapping of duplicate names to their canonical forms.

    Returns:
        pd.DataFrame: A dataframe with the normalized recommendation columns.
        pd.DataFrame: A dataframe with the frequency for each unique response in the top 5.
    
    Side Effect:
        Writes both dataframes to Excel.
    """

    rec_cols = [col for col in answers_df.columns if col.startswith("recommendation_")]
    for col in rec_cols:
        answers_df[col] = answers_df[col].map(normalisation_mapping).fillna(answers_df[col])

    melted = answers_df[rec_cols].melt(value_name="response", value_vars=rec_cols)
    responses_series = melted["response"]
    counts_df = responses_series.value_counts().rename_axis("response").reset_index(name="count")

    with pd.ExcelWriter(f"responses_{model}/{topic}/answers_{topic}.xlsx", engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        answers_df.to_excel(writer, sheet_name="Normalised Data", index=False)
        counts_df.to_excel(writer, sheet_name="Unique Responses Counts", index=False)

    return answers_df, counts_df
