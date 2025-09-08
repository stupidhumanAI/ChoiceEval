import ast
import numpy as np
import pandas as pd
import math
import openai
from openai import OpenAI
from pydantic import BaseModel
from typing import Optional, Type
import os
from google import genai
import openpyxl
import json
from pathlib import Path
from typing import List
from dotenv import load_dotenv

load_dotenv()

topic = "laptops"
nr_of_questions = 9

client = OpenAI(api_key= os.getenv("OPENAI_API_KEY"))
client_gemini = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def gpt_call(user_prompt: str, system_prompt: Optional[str] = None, model_name = "gpt-4o", output_text_format: Optional[Type[BaseModel]] = None):
    """
    Send a prompt to the OpenAI API and return the model's response.

    Args:
        user_prompt (str): The user-provided input prompt.
        system_prompt (Optional[str], default=None): An optional system-level instruction to guide the model's behavior.
        model_name (str, default="gpt-4o"): The name of the OpenAI model to use.
        output_text_format (Optional[Type[BaseModel]], default=None): A Pydantic BaseModel defining the desired output schema. If provided, the response will be parsed into this format.

    Returns:
        str: The generated response text from the model.
    """
    if output_text_format:
       response = client.responses.parse(
                    model=model_name,
                    input=user_prompt,
                    instructions=system_prompt,
                    text_format=output_text_format,
                )
    else:
        response = client.responses.create(
                    model=model_name,
                    input=user_prompt,
                    instructions=system_prompt,
                )

    response_text = response.output[0].content[0].text
    return response_text

def gemini_call(user_prompt: str, model_name = "gemini-2.5-flash"):
    """
    Send a prompt to the Gemini API and return the model's response.

    Args:
        user_prompt (str): The user-provided input prompt.
        model_name (str, default="gemini-2.5-flash"): The name of the Gemini model to use.

    Returns:
        str: The generated response text from the model.
    """
    response = client_gemini.models.generate_content(
        model=model_name,
        contents=user_prompt,
    )
    return response.text

folder_path = Path(f"responses/{topic}")
folder_path.mkdir(parents=True, exist_ok=True)

def translate_clusters(topic: str):
    """
    Translate predefined consumer clusters into topic-specific clusters.

    This function uses GPT to reinterpret a fixed set of consumer groups in the 
    context of a given topic. The output is structured JSON, parsed into the 
    `ConsumerClusters` model.

    Args:
        topic (str): The subject or domain to translate the consumer clusters into.

    Returns:
        ConsumerClusters: An object containing both the original clusters and 
        their topic-specific translations.
    """

    class ConsumerClusters(BaseModel):
        original_consumer_cluster: list[str]
        translated_consumer_clusters: list[str]

    user_prompt_consumer_clusters = f"""
    Consumer groups:
    1. Performance and Quality Seekers
    2. Budget-Conscious Shoppers
    3. Innovation and Technology Enthusiasts
    4. Health and Wellness Focus
    5. Ethical and Environmental Considerations
    6. Convenience-Oriented Shoppers
    7. Experience and Lifestyle Seekers
    8. Novice vs. Expert Levels
    9. Cultural and Social Influences

    Topic: {topic}"""

    system_prompt_consumer_clusters = """You will be given a topic and a list of consumer clusters. Your task is to convert what each cluster should translate into for the specific topic. 
    Output: the list of the original consumer clusters and the translated one for the specific topic. Provide the answer in a structured json in the following format:
    {
    "original_consumer_cluster": list of clusters
    "translated_consumer_clusters": list of translated clusters
    }"""

    new_clusters = ast.literal_eval(gpt_call(user_prompt_consumer_clusters, system_prompt_consumer_clusters, output_text_format = ConsumerClusters))
   
    return new_clusters

new_clusters = translate_clusters(topic)

def generate_questions(topic: str, clusters: List[str]):

    """
    Generate consumer-journey consideration questions per cluster for a given topic. The number of questions is distributed evenly across
    clusters and rounded up where needed.

    Args:
        topic (str): The subject area the questions should target and explicitly mention.
        clusters (List[str]): The list of consumer groups to tailor questions for.
        total_questions (int): The total number of questions to generate across all clusters.

    Returns:
        Dict[str, List[str]]: A mapping from each consumer group name to a list of questions.
    """

    number_of_relevant_clusters = len(clusters)
    number_of_questions_per_cluster = math.ceil(nr_of_questions/number_of_relevant_clusters)

    system_prompt_questions = f"""You will be given a topic, a list of consumer groups and a number of questions. Your task is to create user queries as if they were directed to an AI assistant on a specific topic. The goal is to produce questions that will result in answers which are specific examples of [topic]. 

    For each consumer group, create a list of search queries that a user seeking recommendations in the consideration phase of their consumer journey—belonging to that specific consumer group—might ask. The questions should be formulated so that the answers are instances of the [topic]. The number of questions for each group must be equal to the number you are given.
    Important: each question should specifically mention the [topic] in it. 

    In the consideration phase of the consumer journey, consumers actively explore and evaluate various products or services to address their needs or solve a problem. Key characteristics: research and information gathering, evaluation of alternatives, engagement with brand content, influence of social proof and reviews, development of preferences and shortlists and  establishing expectations and criteria. Ensure each question is asked in a way that the answer would recommend a specific recommendation.

    Example with Topic as Fruits:

    Right Responses (Questions where the answers are specific fruits):

    •	“What fruits are high in vitamin C?”
    •	“Which fruits are best for making smoothies?”
    •	“What are some exotic fruits to try this summer?”
    •	“Which fruits are low in sugar but high in fiber?”

    Wrong Responses (Questions leading to advice or methods):

    •	“How can I ripen fruits faster at home?”
    •	“What is the best way to store different types of fruits?”
    •	“How do I know if a fruit is organic?”
    •	“What are the health benefits of eating fruits daily?”

    Provide the answer in the form of a dictionary in the following structure: """ + """
    {
    "consumer_group_name": [
        "Question 1",
        "Question 2",
        "...",
        "Question N"
    ]
    }
    Do not append or prepend any text, return it in this exact form.
    """

    user_prompt_questions = f"""
    Topic: {topic}
    Consumer grpups: {new_clusters['translated_consumer_clusters']}
    Number of questions: {number_of_questions_per_cluster}"""

    questions_json = ast.literal_eval(gpt_call(user_prompt_questions, system_prompt_questions))
    return questions_json


questions_json = generate_questions(topic, new_clusters['translated_consumer_clusters'])

def list_entities(text: str, topic: str):
    """
    Extract up to five {topic} brands or providers from a text, in order of appearance.

    Args:
        text (str): The source text to analyze.
        topic (str): The domain label that contextualizes the entities (e.g., "laptops",
            "insurance providers").

    Returns:
        Dict[str, str]: A dictionary with keys "recommendation_1" through
        "recommendation_5" mapped to the extracted brand/provider names (or
        "No recommendation" where applicable).
    """
    class Recommendations(BaseModel):
        recommendation_1: str
        recommendation_2: str
        recommendation_3: str
        recommendation_4: str
        recommendation_5: str

    system_prompt_entities = f"""You will be given a text about {topic}.
        Act as 5 independent experts. Analyze the text and list first 5 {topic} brands or providers
        recommended in the text. If a specific product or service is mentioned,
        list the brand or provider, not the product or service itself.
        Extremely important: List the {topic} brands or providers in the exact order they
        are mentioned in the text.
        Provide output a simple majority agrees upon.

        Provide output in a dictionary with the following format: """ + """
        {
         "recommendation_1": recommendation 1,
         "recommendation_2": recommendation 2,
         "recommendation_3": recommendation 3,
         "recommendation_4": recommendation 4,
         "recommendation_5": recommendation 5,
        }
        """ + f"""
        If less then 5 {topic} brands or providers are mentioned, return 'No recommnedation' for the missing recommendations.
        If no {topic} brands or providers are mentioned, return 'No recommnedation' in place of all 5 recommendations.
        """
    entities_dictionary = ast.literal_eval(gpt_call(text, system_prompt_entities, output_text_format = Recommendations))
    return entities_dictionary
    

def create_interview_answers(topic: str, questions: dict[str, list[str]]):
    """
    Generate a Q&A dataset by asking each question to two LLMs (GPT and Gemini),
    capturing answers, and extracting up to five ordered recommendations per answer.

    For every input question, the function:
      1) Calls `gpt_call(question)` and `gemini_call(question)`.
      2) Stores the raw answer, the LLM identifier ("gpt" or "gemini"), and the topic.
      3) Extracts entities via `list_entities(answer, topic)` and fills
         `recommendation_1` … `recommendation_5`.

    The output is a tidy DataFrame with one row per (question model)and , i.e., two rows
    per question, containing: consumer cluster, question, topic, llm, answer, and
    five recommendation columns. The DataFrame is also saved to an Excel file
    at `responses/{topic}/answers_{topic}.xlsx` (sheet: "Raw Data").

    Args:
        topic (str): The subject domain of the interview (e.g., "laptops").
        questions (dict[str, list[str]]): A mapping of consumer-cluster names to the
            list of questions to ask for that cluster.

    Returns:
        pandas.DataFrame: A DataFrame with columns:
            - consumer_cluster (str)
            - question (str)
            - topic (str)
            - llm (str: "gpt" or "gemini")
            - answer (str)
            - recommendation_1 … recommendation_5 (str)

    
    """
    answers_df = pd.DataFrame([(key, value) for key, values in questions.items() for value in values], columns=['consumer_cluster', 'question'])
    nr_of_questions = len(answers_df)
    answers_df = pd.concat([answers_df, answers_df], ignore_index=True)

    answers_df["topic"] = topic
    answers_df["llm"] = None
    answers_df["answer"] = None
    answers_df["recommendation_1"] = None
    answers_df["recommendation_2"] = None
    answers_df["recommendation_3"] = None
    answers_df["recommendation_4"] = None
    answers_df["recommendation_5"] = None

    for index, row in answers_df[:nr_of_questions].iterrows():
            interview_question = row['question']

            interview_response_gpt = gpt_call(interview_question)
            answers_df.at[index, 'llm'] = "gpt"
            answers_df.at[index, 'answer'] = interview_response_gpt
            entities_gpt = list_entities(interview_response_gpt, topic)
            answers_df.loc[index, entities_gpt.keys()] = pd.Series(entities_gpt)

            gemini_index = index + nr_of_questions
            interview_response_gemini = gemini_call(interview_question)
            answers_df.at[gemini_index, 'llm'] = "gemini"
            answers_df.at[gemini_index, 'answer'] = interview_response_gemini
            entities_gemini = list_entities(interview_response_gemini, topic)
            answers_df.loc[gemini_index, entities_gemini.keys()] = pd.Series(entities_gemini)

            answers_df.to_excel(f"responses/{topic}/answers_{topic}.xlsx", sheet_name="Raw Data", index=False)
    return answers_df

answers_df = create_interview_answers(topic, questions_json)

def normalise_responses(answers_df: pd.DataFrame, topic: str, max_pairs_to_compare: int = 15):
    """
    Detect and consolidate duplicate brand/provider names across recommendation columns. When the model flags a duplicate, the longer string
    (`duplicated_value`) is mapped to the shorter canonical form (`original_value`).
    The mapping is applied to the recommendations and saved to disk.

    Args:
        answers_df (pd.DataFrame): Input dataframe containing recommendations.
        topic (str): The subject domain of the interview (e.g., "laptops").
        max_pairs_to_compare (int, optional): For each candidate value, compare against
            up to this many higher-frequency values (reduces API calls). Defaults to 15.

    Returns:
        Normalised pd.DataFrame

    Side Effect:
        - Writes `normalisation.json` into reponses folder.
    """
    class NormalisationDictionary(BaseModel):
        duplicated_value: str
        original_value: str 

    rec_cols = [col for col in answers_df.columns if col.startswith("recommendation_")]
    all_recs = answers_df[rec_cols].values.ravel()     
    counts = pd.Series(all_recs).value_counts()
    counts_df = counts.reset_index()
    counts_df.columns = ["value", "count"]

    system_prompt_normalisation = f"""You will receive 2 {topic} brands or providers. 
    Your task is to check if any brand or provider appears to be a duplicate of another. 

    If duplicates exist:
    - Return them in JSON format.
    - The "duplicated_value" should be the longer longer.
    - The "original_value" should always be the shorter string. 
    If no duplicates are found, return a single JSON object with both values set to None. 
    If any of the value is 'No recommnedation', return a single JSON object with both values set to None. """ + """

    Output strictly in this JSON structure:

    {
        "duplicated_value": "duplicated_value",
        "original_value": "original_value"
    }
    """

    normalisation_mapping = {}
    normalised_counts_df = counts_df.copy()

    def apply_mapping_if_valid(d: dict):
        dv = d.get("duplicated_value")
        ov = d.get("original_value")
        if dv and ov and dv != ov and dv != 'No recommnedation':
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
        if (pd.isna(vi) or vi.strip() == "" or vi == "No recommnedation"):
            continue
        for j in range(0, min(i,15)):
            vj = normalised_counts_df.iloc[j]["value"]
            print(f"vi: {vi}, vj: {vj}")
            user_prompt_normalisation = vi + ", " + vj
            mapping_dictionary = ast.literal_eval(gpt_call(user_prompt=user_prompt_normalisation, system_prompt=system_prompt_normalisation, output_text_format=NormalisationDictionary))
            apply_mapping_if_valid(mapping_dictionary)
            normalised_counts_df = normalised_counts_df.groupby('value', as_index=False, sort=False).agg(count=('count', 'sum')).sort_values(by="count", ascending=False).reset_index(drop=True)
            if vi in normalisation_mapping:
                break

    normalisation_json_path = f"responses/{topic}/normalisation.json"
    with open(normalisation_json_path, "w") as f:
        json.dump(normalisation_mapping, f, indent=4)


    rec_cols = [col for col in answers_df.columns if col.startswith("recommendation_")]
    for col in rec_cols:
        answers_df[col] = answers_df[col].map(normalisation_mapping).fillna(answers_df[col])

    with pd.ExcelWriter(f"responses/{topic}/answers_{topic}.xlsx", engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
            answers_df.to_excel(writer, sheet_name="Normalised Data", index=False)
    return answers_df

answers_df = normalise_responses(answers_df, topic)

def analyse_responses(answers_df: pd.DataFrame):
    answers_df_clean = answers_df.melt(
        id_vars=["consumer_cluster", "topic", "llm"],         
        value_vars=[f"recommendation_{i}" for i in range(1, 6)],  
        var_name="recommendation_rank",         
        value_name="recommendation_value"     
    )

    answers_df_clean["recommendation_rank"] = answers_df_clean["recommendation_rank"].str.extract(r"(\d+)")
    answers_df_clean = answers_df_clean.dropna(subset=["recommendation_value"])           
    answers_df_clean = answers_df_clean[answers_df_clean["recommendation_value"].str.strip() != ""]  

    unique_brands = answers_df_clean["recommendation_value"].unique()
    geo_mapping = {}

    location_system_prompt = """You will be given a brand, company or other entity. Your task is to determine, which geogrpahical location is this entity assosiacted with.
    Select one of the following: USA, Canada, Europe, Asia, Other or None if not applicable. Return only single word denoting the geo location, nothing else. 
    """

    for brand in unique_brands:
        if pd.notna(brand) and brand != "No recommnedation":  
            answer_geo_location = gpt_call(brand, location_system_prompt)
            geo_mapping[brand] = answer_geo_location
    answers_df_clean["geo_location"] = answers_df_clean["recommendation_value"].map(geo_mapping)

    with pd.ExcelWriter(f"responses/{topic}/answers_{topic}.xlsx", engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
            answers_df_clean.to_excel(writer, sheet_name="Data for Analysis", index=False)

    answers_df_clean = pd.read_excel(f"responses/{topic}/answers_{topic}.xlsx", sheet_name="Data for Analysis")

    total_preference = (
        answers_df_clean.groupby(["llm", "recommendation_value"])
        .size()
        .reset_index(name="count")
        .assign(  
            sort_key=lambda df: (df["recommendation_value"] != "No recommendation").astype(int)
        )
        .sort_values(
            by=["llm", "sort_key", "count"], 
            ascending=[False, True, False] 
        )
        .drop(columns="sort_key")  
        .reset_index(drop=True)
    )

    denoms = (
        answers_df_clean.loc[
            (answers_df_clean["recommendation_rank"] == 1) &
            (answers_df_clean["recommendation_value"] != "No recommendation")
        ]
        .groupby("llm")
        .size()
        .rename("denom")
    )


    total_preference = (
        total_preference
        .assign(denom=lambda df: df["llm"].map(denoms))
        .assign(share=lambda df: df["count"] / df["denom"])
        .drop(columns="denom")
    )

    total_preference.loc[
        total_preference["recommendation_value"] == "No recommendation", "share"
    ] = None

    with pd.ExcelWriter(f"responses/{topic}/answers_{topic}.xlsx", engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
            total_preference.to_excel(writer, sheet_name="Recommendations", index=False)

    geo_preference = (
        answers_df_clean.groupby(["llm", "consumer_cluster", "geo_location"])
        .size()
        .reset_index(name="count")
        .sort_values(["llm", "consumer_cluster", "geo_location"])
        .reset_index(drop=True)
    )

    geo_wide = (
        geo_preference.pivot(
            index=["llm", "consumer_cluster"],  # rows
            columns="geo_location",             # new columns
            values="count"                      # fill with counts
        )
        .fillna(0)   # replace NaN with 0 if some llm/cluster has no rows in that geo
        .reset_index()
    )

    possible_cols = ["USA", "Canada", "Europe", "Asia", "Other"]
    cols_to_sum = [c for c in possible_cols if c in geo_wide.columns]
    geo_wide["Total"] = geo_wide[cols_to_sum].sum(axis=1)

    for col in cols_to_sum:
        geo_wide[f"{col}_share"] = geo_wide[col] / geo_wide["Total"]

    if "USA" in geo_wide.columns and "Europe" in geo_wide.columns:
        geo_wide["LOR_US_Europe"] = np.log((geo_wide["USA"] + 0.5) / (geo_wide["Europe"] + 0.5))
    if "USA" in geo_wide.columns and "Asia" in geo_wide.columns:
        geo_wide["LOR_US_Asia"] = np.log((geo_wide["USA"] + 0.5) / (geo_wide["Asia"] + 0.5))
    if "USA" in geo_wide.columns and "Other" in geo_wide.columns:
        geo_wide["LOR_US_Other"] = np.log((geo_wide["USA"] + 0.5) / (geo_wide["Other"] + 0.5))

    with pd.ExcelWriter(f"responses/{topic}/answers_{topic}.xlsx", engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
            geo_wide.to_excel(writer, sheet_name="Statistical Variables", index=False)

analyse_responses(answers_df)