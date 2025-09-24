import json
import os
from pathlib import Path
from dotenv import load_dotenv

from generate_dataset import generate_questions
from process_responses import (
    assign_location, 
    create_interview_answers, 
    normalise_responses, 
    apply_normalisation, 
    analyse_responses
)

load_dotenv()

topic = "airlines"
model = "deepseek"
use_already_existing_questions = False
nr_of_questions = 207

folder_path = Path(f"responses/{topic}")
folder_path.mkdir(parents=True, exist_ok=True)

if use_already_existing_questions:
    path = f"questions/{topic}.json"
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    
    with open(path, "r") as file:
        questions_json = json.load(file)
else: 
    questions_json = generate_questions(topic, total_questions=nr_of_questions)

answers_df = create_interview_answers(topic, model, questions_json)
normalisation_mapping = normalise_responses(answers_df, topic)
processed_answers_df, counts_df = apply_normalisation(answers_df, topic, normalisation_mapping)
geo_tagged_counts_df = assign_location(counts_df, topic)
analyse_responses(processed_answers_df, geo_tagged_counts_df, topic)