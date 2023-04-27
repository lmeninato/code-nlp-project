import torch
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from data import load_local_dataset
from finetune_data import custom_dataloader
from sacrebleu import corpus_bleu

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")
model.eval()

dataset = load_local_dataset("python", "data")

train_data_loader = custom_dataloader(
    dataset["train"],
    tokenizer,
    512,
    512,
    16,
    4,
)


def generate_docstrings(code_samples: torch.Tensor, tokenizer, model):
    input_texts = [
        f"Generate a docstring for the following code: {code}"
        for code in tokenizer.batch_decode(code_samples, skip_special_tokens=True)
    ]

    input_ids = tokenizer(
        input_texts, truncation=True, return_tensors="pt", padding=True, max_length=512
    ).input_ids

    with torch.no_grad():
        outputs = model.generate(input_ids, max_length=512)

    return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]


# Load the dataset and create a DataLoader for the validation set
dataset = load_local_dataset("python", "data")
validation_data_loader = custom_dataloader(
    dataset["validation"],
    tokenizer,
    512,
    512,
    16,
    4,
)


for code, docstring in validation_data_loader:
    docstrings = generate_docstrings(code, tokenizer, model)

    reference_docstring = [
        tokenizer.decode(ds, skip_special_tokens=True) for ds in docstring
    ]

    bleu_scores = []

    for docstring, ref in zip(docstrings, reference_docstring):
        score = corpus_bleu(docstring, [ref]).score
        print(f"Got bleu score: {score}")
        bleu_scores.append(score)

    print(bleu_scores)
    break
