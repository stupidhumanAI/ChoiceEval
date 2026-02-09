import json
import os
from dotenv import load_dotenv

from generate_dataset import generate_questions
from process_responses import (
    create_interview_answers, 
    normalise_responses, 
    apply_normalisation, 
)
from statistical_tests import assign_location, analyse_responses

load_dotenv()

# ---- Experiment configuration (edit these) ----
topic = "airlines"
normalisation_grouping = "brands"
model = "deepseek"

# Question generation controls
use_already_existing_questions = False
nr_of_questions = 207

# Analysis controls
test_geographic_bias = True

# 1) Get questions (either load from disk or generate)
if use_already_existing_questions:
    path = f"questions/{topic}.json"
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    
    with open(path, "r") as file:
        questions_json = json.load(file)
else: 
    questions_json = generate_questions(topic, total_questions=nr_of_questions)

# 2) Collect answers + extract top-5 recommendations
answers_df = create_interview_answers(topic, model, normalisation_grouping, questions_json)

# 3) Normalize duplicates (e.g., product -> brand)
normalisation_mapping = normalise_responses(answers_df, topic, model, normalisation_grouping)
processed_answers_df, counts_df = apply_normalisation(answers_df, topic, model, normalisation_mapping)

# 4) Optional geographic bias analysis
if test_geographic_bias: 
    geo_tagged_counts_df = assign_location(counts_df, topic, model)
    analyse_responses(processed_answers_df, geo_tagged_counts_df, topic, model)
