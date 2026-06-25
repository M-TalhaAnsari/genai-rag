from langsmith import Client
from langsmith.evaluation import evaluate
from dotenv import load_dotenv
load_dotenv()

client = Client()

# Dataset name
dataset_name = "Sample Dataset"

# Create dataset
dataset = client.create_dataset(
    dataset_name,
    description="A sample dataset"
)

# Add examples
client.create_examples(
    inputs=[
        {
            "postfix": "Yoga offers numerous benefits for both the body and mind."
        }
    ],
    outputs=[
        {
            "output": "The advantages of practicing yoga are extensive."
        }
    ],
    dataset_id=dataset.id,
)

# Custom evaluator
def exact_match(run, example):
    prediction = run.outputs["output"]
    expected = example.outputs["output"]

    return {
        "key": "exact_match",
        "score": prediction == expected
    }

# Run evaluation
experiment_results = evaluate(
    lambda inputs: {"output": inputs["postfix"]},
    data=dataset_name,
    evaluators=[exact_match],
    experiment_prefix="sample-experiment",
    metadata={
        "version": "1.0.0",
        "revision_id": "beta"
    }
)

print(experiment_results.experiment_name)