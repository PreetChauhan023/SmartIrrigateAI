"""
Generate 300 synthetic sugarcane guidance records.

Run:
    python data/generate_guidance.py

All records are fictional educational examples.
They are NOT validated agricultural recommendations.
"""

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "knowledge" / "synthetic_guidance.json"

random.seed(42)

TOPICS = {
    "Sugarcane Irrigation Management": {
        "concepts": [
            ("irrigation scheduling", "Combine soil observations, crop stage and weather forecasts before deciding when to irrigate."),
            ("irrigation frequency", "Irrigation frequency depends on soil water retention, crop demand, rainfall and local conditions."),
            ("irrigation duration", "Duration estimates depend on water delivery rate, soil conditions and field requirements."),
            ("irrigation efficiency", "Compare water applied with observed soil moisture changes and crop response."),
            ("irrigation records", "Record irrigation dates, duration, estimated water volume and field observations."),
        ],
        "contexts": [
            "a newly planted sugarcane field",
            "a sugarcane field during active growth",
            "a mature sugarcane crop",
            "a field with uneven soil moisture",
            "a field using drip irrigation",
            "a field using furrow irrigation",
        ],
    },
    "Soil Moisture and Soil Health": {
        "concepts": [
            ("soil moisture monitoring", "Use calibrated measurements from representative field locations rather than relying on one sensor."),
            ("soil texture", "Sandy and clay-rich soils differ in infiltration, drainage and water retention."),
            ("soil water retention", "Evaluate how effectively the soil stores water available to crop roots."),
            ("sensor calibration", "Check sensor calibration, measurement units and placement before trusting readings."),
            ("soil drainage", "Inspect waterlogging, infiltration and drainage conditions before applying additional water."),
        ],
        "contexts": [
            "sandy soil",
            "clay-rich soil",
            "loamy soil",
            "a field with variable soil texture",
            "a field with newly installed sensors",
            "a field after heavy rainfall",
        ],
    },
    "Weather and Rainfall": {
        "concepts": [
            ("rainfall probability", "A probability of rain is not a guarantee of rainfall or an estimate of its total amount."),
            ("temperature variation", "Temperature influences evaporative demand but must be interpreted alongside other weather variables."),
            ("weather forecasts", "Forecasts contain uncertainty and should be updated before operational decisions."),
            ("evapotranspiration", "Evapotranspiration represents water transferred through evaporation and plant transpiration."),
            ("rainfall observations", "Measured rainfall and forecast rainfall should be recorded separately."),
        ],
        "contexts": [
            "a hot afternoon",
            "a cloudy day",
            "a forecast of possible rain",
            "a week of dry weather",
            "a sudden temperature increase",
            "an unexpected rainfall event",
        ],
    },
    "Crop Growth Stages": {
        "concepts": [
            ("germination stage", "Early crop development requires monitoring establishment and root-zone conditions."),
            ("tillering stage", "Track crop development and soil moisture while evaluating irrigation requirements."),
            ("grand growth stage", "Rapid crop development can change water demand and should be reflected in model inputs."),
            ("maturity stage", "Late-stage irrigation decisions require crop-specific and locally validated guidance."),
            ("crop age recording", "Record planting dates and growth-stage observations consistently."),
        ],
        "contexts": [
            "a recently planted crop",
            "a three-month-old crop",
            "a five-month-old crop",
            "a seven-month-old crop",
            "a ten-month-old crop",
            "a field with uneven crop development",
        ],
    },
    "Water Conservation": {
        "concepts": [
            ("water-use efficiency", "Measure water consumption and compare it with crop and soil observations."),
            ("drip irrigation", "Drip systems deliver water locally and require maintenance and flow monitoring."),
            ("irrigation losses", "Water may be lost through runoff, evaporation, leakage or excessive application."),
            ("water budgeting", "Track available water, planned irrigation and actual consumption."),
            ("water conservation monitoring", "Compare historical water use with current observations before claiming savings."),
        ],
        "contexts": [
            "a water-limited farm",
            "a drip irrigation system",
            "a furrow irrigation system",
            "a field with high evaporation",
            "a farm monitoring water consumption",
            "a farm comparing irrigation methods",
        ],
    },
    "Machine Learning Predictions": {
        "concepts": [
            ("prediction confidence", "Model confidence is not a guarantee of correctness or field safety."),
            ("classification predictions", "A classification model can estimate whether irrigation may be needed."),
            ("regression predictions", "A regression model can estimate irrigation duration from input features."),
            ("feature importance", "Feature importance helps describe model behavior but does not establish causality."),
            ("model evaluation", "Evaluate models on held-out data using suitable classification and regression metrics."),
        ],
        "contexts": [
            "a decision tree model",
            "a random forest model",
            "a gradient boosting model",
            "a model trained on synthetic data",
            "a newly deployed model",
            "a model receiving unfamiliar inputs",
        ],
    },
    "MLOps Monitoring and Data Drift": {
        "concepts": [
            ("data drift", "Data drift occurs when incoming feature distributions differ from the training reference distribution."),
            ("prediction drift", "Changes in prediction distributions can indicate changing inputs or model behavior."),
            ("model versioning", "Record the model version used for each prediction to support traceability."),
            ("drift detection", "Compare reference and incoming feature distributions using appropriate statistical measures."),
            ("model retraining", "Investigate drift and evaluate a candidate model before approving retraining or deployment."),
        ],
        "contexts": [
            "a sudden temperature shift",
            "a change in soil moisture patterns",
            "a new growing season",
            "a recently updated model",
            "an unusual rainfall pattern",
            "a monitoring dashboard alert",
        ],
    },
    "Agricultural Troubleshooting": {
        "concepts": [
            ("unexpected model predictions", "Check input values, model version and measurement quality before interpreting unusual predictions."),
            ("missing sensor readings", "Identify missing values and use validated handling rules rather than inventing measurements."),
            ("unusual soil moisture readings", "Verify sensor calibration, location and recent weather observations."),
            ("inconsistent weather inputs", "Check forecast timestamps, measurement units and data sources."),
            ("prediction disagreements", "Record expert-reviewed corrections and investigate systematic prediction errors."),
        ],
        "contexts": [
            "a sensor returning zero",
            "a sensor producing unusually high values",
            "a missing weather forecast",
            "conflicting field observations",
            "a prediction that appears incorrect",
            "a field with incomplete records",
        ],
    },
    "LangChain and RAG": {
        "concepts": [
            ("retrieval-augmented generation", "RAG retrieves relevant documents and provides them as context to a language model."),
            ("semantic search", "Embedding models represent text numerically so semantically related passages can be retrieved."),
            ("vector databases", "Vector stores index embeddings to support similarity-based document retrieval."),
            ("source citations", "Citations should identify the retrieved records supporting an answer."),
            ("hallucination prevention", "Ground answers in retrieved context and explicitly acknowledge missing evidence."),
        ],
        "contexts": [
            "an agricultural chatbot",
            "a multilingual assistant",
            "a synthetic knowledge base",
            "a farmer asking follow-up questions",
            "a question requiring several documents",
            "a question outside the knowledge base",
        ],
    },
    "Feedback and Model Improvement": {
        "concepts": [
            ("human feedback", "Collect corrections together with original inputs, predictions and model versions."),
            ("feedback verification", "Expert review is necessary before treating user corrections as reliable labels."),
            ("retraining datasets", "Use validated examples and preserve separate evaluation data."),
            ("model comparison", "Compare candidate models using identical evaluation partitions and metrics."),
            ("deployment approval", "Deploy a new model only after validation and approval checks."),
        ],
        "contexts": [
            "a farmer rejecting a prediction",
            "an agronomist reviewing feedback",
            "a candidate model update",
            "a model performing poorly",
            "a new feedback dataset",
            "a scheduled model evaluation",
        ],
    },
}

QUESTION_TEMPLATES = [
    "Why is {concept} important for {context}?",
    "How should {concept} be evaluated in {context}?",
    "What should be considered about {concept} in {context}?",
    "How can a monitoring system handle {concept} in {context}?",
    "What are the limitations of {concept} for {context}?",
    "How can {concept} support decisions involving {context}?",
]


def generate():
    records = []

    for category, configuration in TOPICS.items():
        concepts = configuration["concepts"]
        contexts = configuration["contexts"]

        for concept_index, (concept, guidance) in enumerate(concepts):
            for context_index, context in enumerate(contexts):
                template = QUESTION_TEMPLATES[
                    (concept_index + context_index) % len(QUESTION_TEMPLATES)
                ]

                question = template.format(
                    concept=concept,
                    context=context,
                )

                answer = (
                    f"{guidance} "
                    f"In the example of {context}, first verify the available "
                    f"measurements and their reliability. "
                    f"Record the relevant observations and compare them with "
                    f"previous measurements before interpreting changes. "
                    f"For this educational demonstration, the concept of "
                    f"{concept} illustrates how an intelligent irrigation "
                    f"system can support decisions while retaining human review. "
                    f"These examples are synthetic and should not be treated "
                    f"as validated agricultural prescriptions."
                )

                records.append({
                    "id": f"DEMO-{len(records) + 1:03d}",
                    "category": category,
                    "topic": concept,
                    "context": context,
                    "question": question,
                    "answer": answer,
                    "keywords": [
                        concept,
                        category,
                        context,
                        "sugarcane",
                        "irrigation",
                    ],
                    "source": "Synthetic educational knowledge base",
                    "synthetic": True,
                })

    assert len(records) == 300

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    OUTPUT.write_text(
        json.dumps(records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Generated {len(records)} synthetic knowledge records.")
    print(f"Saved to: {OUTPUT}")


if __name__ == "__main__":
    generate()