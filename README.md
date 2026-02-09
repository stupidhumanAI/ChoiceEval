# Auditing Preferences for Brands and Cultures in LLMs

Generate and analyze LLM recommendation preferences by consumer cluster. The pipeline creates topic-specific questions, collects model answers, extracts top recommendations, normalizes duplicates, and runs geographic bias analysis.

## Paper Context

This repository contains the code linked to our paper, **"Auditing Preferences for Brands and Cultures in LLMs."**

- We design **ChoiceEval**, a comprehensive framework for generating diverse evaluation questions, grounded in established consumer segmentation research.
- The framework enables researchers to generate contextually relevant evaluation questions for any topic of interest, ensuring broader applicability.

## What It Does

- Translates generic consumer clusters into topic-specific clusters.
- Generates consideration-phase questions per cluster.
- Calls an LLM to answer each question.
- Extracts the first 5 recommendations from each answer.
- Normalizes duplicates (e.g., product -> brand).
- Optionally maps recommendations to regions and computes log-odds ratios.

## Setup

1. Create a virtual environment (optional but recommended).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file with API keys:

```env
OPENAI_API_KEY=...
GEMINI_API_KEY=...
DEEPSEEK_API_KEY=...
```

## Run

Edit `main.py` to set your experiment parameters:

- `topic`: domain to study (e.g., `"airlines"`).
- `normalisation_grouping`: grouping to normalize to (e.g., `"brands"`, `"country"`, `"city"`).
- `model`: `"gpt"`, `"gemini"`, or `"deepseek"`.
- `use_already_existing_questions`: reuse `questions/{topic}.json` if true.
- `nr_of_questions`: total questions to generate (distributed across clusters).
- `test_geographic_bias`: enable region assignment and log-odds analysis.

Then run:

```bash
python main.py
```

## Outputs

Results are written to `responses_{model}/{topic}/answers_{topic}.xlsx` with sheets:

- `Raw Data`: Questions, answers, extracted recommendations.
- `Normalised Data`: Recommendations after normalization mapping.
- `Unique Responses Counts`: Frequency table of recommendations.
- `Data for Analysis`: Melted data with geo mapping (if enabled).
- `Statistical Variables`: Geo preference shares and log-odds (if enabled).

Normalization mapping is saved to:

- `responses_{model}/{topic}/normalisation.json`

## Verification

- Review and verify the normalization mapping in `responses_{model}/{topic}/normalisation.json` before relying on downstream analysis.

## Repo Structure

- `main.py`: Orchestrates the full pipeline.
- `generate_dataset.py`: Cluster translation and question generation.
- `process_responses.py`: Answer collection, extraction, normalization.
- `statistical_tests.py`: Geographic mapping and analysis.
- `utils.py`: API clients for OpenAI, Gemini, DeepSeek.
- `questions/`: Optional cached question sets by topic.
- `responses_{model}/`: Generated outputs.

## Notes

- The first run can be long depending on `nr_of_questions`.
- If `answers_{topic}.xlsx` already exists, unanswered rows are resumed instead of re-run.
- Geographic assignment uses the same LLM to map each recommendation to a region.
