import ast
import math
from dotenv import load_dotenv
from pydantic import BaseModel
from utils import gpt_call

load_dotenv()


def translate_clusters(topic: str) -> list[str]:
    """
    Translate predefined consumer clusters into topic-specific clusters.
    This function uses GPT to reinterpret a fixed set of consumer groups in the context of a given topic.

    Args:
        topic (str): The subject or domain to translate the consumer clusters into.

    Returns:
        ConsumerCluster: A object containing the list of the translated topic-specific clusters.
    """

    user_prompt_consumer_clusters = f"""Consumer groups:
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

    system_prompt_consumer_clusters = """You will be given a topic and a list of consumer clusters. 
    Your task is to convert what each cluster should translate into for the specific topic. 
    Output: the list of the translated consumer clusters for the specific topic. Provide the answer in a structured json in the following format:
    {
        "translated_consumer_clusters": list of translated clusters
    }"""

    class ConsumerClusters(BaseModel):
        translated_consumer_clusters: list[str]

    new_clusters = gpt_call(user_prompt_consumer_clusters, system_prompt_consumer_clusters, output_text_format = ConsumerClusters)
    return ast.literal_eval(new_clusters)['translated_consumer_clusters']


def generate_questions(topic: str, total_questions: int = 207) -> dict[str, list[str]]:
    """
    Generate consumer-journey consideration questions per cluster for a given topic. The number of questions is distributed evenly across
    clusters and rounded up where needed.

    Args:
        topic (str): The subject area the questions should target and explicitly mention.
        total_questions (int): The total number of questions to generate across all clusters.

    Returns:
        dict[str, list[str]]: A mapping from each consumer group name to a list of questions.
    """
    clusters = translate_clusters(topic)
    n_questions_per_cluster = math.ceil(total_questions/len(clusters))

    system_prompt_questions = """You will be given a topic, a list of consumer groups and a number of questions. Your task is
    to create user queries as if they were directed to an AI assistant on a specific topic. The goal is to produce questions
    that will result in answers which are specific examples of [topic].

    For each consumer group, create a list of search queries that a user seeking recommendations in the consideration phase of
    their consumer journey--belonging to that specific consumer group--might ask. The questions should be formulated so that the
    answers are instances of the [topic]. The number of questions for each group must be equal to the number you are given.
    Important: each question should specifically mention the [topic] in it.

    In the consideration phase of the consumer journey, consumers actively explore and evaluate various products or services
    to address their needs or solve a problem. Key characteristics: research and information gathering, evaluation of alternatives,
    engagement with brand content, influence of social proof and reviews, development of preferences and shortlists and establishing
    expectations and criteria. Ensure each question is asked in a way that the answer would recommend a specific recommendation.

    Example with Topic as Fruits:

    Right Responses (Questions where the answers are specific fruits):

    - "What fruits are high in vitamin C?"
    - "Which fruits are best for making smoothies?"
    - "What are some exotic fruits to try this summer?"
    - "Which fruits are low in sugar but high in fiber?"

    Wrong Responses (Questions leading to advice or methods):

    - "How can I ripen fruits faster at home?"
    - "What is the best way to store different types of fruits?"
    - "How do I know if a fruit is organic?"
    - "What are the health benefits of eating fruits daily?"

    Provide a string containing a dictionary in the following structure:
    {
        "consumer_group_name": [
            "Question 1",
            "Question 2",
            "...",
            "Question N"
        ]
    }
    Do not append or prepend any text, return it in this exact form."""

    user_prompt_questions = f"""Topic: {topic}
Consumer groups: {clusters}
Number of questions: {n_questions_per_cluster}"""

    questions_json = gpt_call(user_prompt_questions, system_prompt_questions)
    return ast.literal_eval(questions_json)