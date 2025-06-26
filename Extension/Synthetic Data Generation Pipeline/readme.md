## The directory contains the synthetic dataset generation pipeline. 

* **sample\_narrations.py:** Reads Ego4D narrations and randomly samples non-overlapping groups of N consecutive clips per video, outputting them as JSON.
* **query\_generator.py:** Loads those sampled narrations and, using Google GenAI plus a prompt guide, generates one past-tense natural-language question per clip group following predefined templates.
* **FarmattingFromIndividualOutputToOneSingleSyntheticDatasetFile.py:** Merges all per-partition query JSONs into a single synthetic NLQ dataset, assigning UUIDs and assembling clip-level annotations into a unified output JSON.
* **llm\_prompt.txt:** Provides the detailed instruction set and 13 question templates that steer the LLM’s generation of grounded, past-tense queries in JSON format.
