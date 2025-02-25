import argparse
import re
from datasets import load_dataset
from tqdm import tqdm
import torch

from ..torch.config import MoondreamConfig
from ..torch.moondream import MoondreamModel
from ..torch.weights import load_weights_into_model

from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_answer(input_text):
    # Find all numbers in the text
    numbers = re.findall(r"\d+", input_text)

    # Return the last number if found, otherwise None
    return numbers[-1] if numbers else None


def eval_mmlu(model, debug=False):
    """Evaluate the model on the MMLU dataset."""
    # Load the hellaswag dataset, validation is 10k rows so only grab 1k random ones for our eval here
    mmlu_test = (
        load_dataset("Rowan/hellaswag", split="validation")
        .shuffle(seed=42)
        .select(range(1000))
    )

    correct = 0
    total = 0
    results = []

    for row in tqdm(mmlu_test, disable=debug, desc="HellaSwag"):
        question = (
            "Return the number of the best ending for this passage.\n\n"
            f'Passage: "{row["activity_label"]}. {row["ctx"]}"\n\n'
            "Endings:\n"
            f'0. {row["endings"][0]}\n'
            f'1. {row["endings"][1]}\n'
            f'2. {row["endings"][2]}\n'
            f'3. {row["endings"][3]}\n'
            f"Your response should be a single number representing the correct ending- 0, 1, 2, or 3."
        )

        # Extract the ground truth answer from the answer field, just the number
        gt_answer = int(row["label"])

        # Encode the question for the model
        model_response = model._text_query(question)["answer"]

        model_answer = parse_answer(model_response)

        # Convert to float for comparison (handling both integers and decimals)
        try:
            if int(model_answer) == int(gt_answer):
                is_correct = True
            else:
                is_correct = False
        except:
            is_correct = False

        if is_correct:
            correct += 1
        total += 1

        result = {
            "question": question,
            "ground_truth": gt_answer,
            "model_response": model_response,
            "model_answer": model_answer,
            "correct": is_correct,
        }
        results.append(result)

        if debug:
            print(f"Question: {question}")
            print(f"Ground Truth Answer: {gt_answer}")
            print(f"Model Response: {model_response}")
            print(f"Model Answer: {model_answer}")
            print(f"Correct: {is_correct}")
            print(f"Current Accuracy: {correct/total:.4f}")
            print("---------")

    accuracy = correct / total if total > 0 else 0

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "results": results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.set_default_device("cuda")
    elif torch.backends.mps.is_available():
        torch.set_default_device("mps")

    config = MoondreamConfig()
    model = MoondreamModel(config)

    load_weights_into_model(args.model, model)

    # Compile omitted to make text only work
    # model.compile()

    result = eval_mmlu(model, args.debug)

    print(f"Accuracy: {result['accuracy']:.4f} ({result['correct']}/{result['total']})")
