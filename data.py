import torch
import glob
from datasets import load_dataset
from pathlib import Path
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from utils import filter_by_token_length


class CodeSearchNetDataset(Dataset):
    """
    CodeSearchNet Dataset.

    This class provides a PyTorch Dataset implementation for the CodeSearchNet dataset.
    It tokenizes and processes the code and docstring samples to be used in a model.

    Usage:

    dataset = load_local_dataset()
    train_data = dataset["train"]
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-2B-mono")
    tokenizer.pad_token = tokenizer.eos_token

    # Filter out examples with more than 512 tokens (or args.max_function_length)
    train_data = train_data.filter(
        lambda example: filter_by_token_length(
            tokenizer, example, args.max_function_length
        )
    )

    train_dataset = CodeSearchNetDataset(
        train_data, tokenizer, max_length=args.max_function_length
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    """

    def __init__(self, data, tokenizer, max_length=512):
        """
        Initialize the dataset with data, tokenizer, and max_length.

        Args:
            data (List[Dict]): A list of dictionaries containing code and docstring samples. # noqa
            tokenizer (PreTrainedTokenizer): The tokenizer to be used for tokenization.
            max_length (int, optional): The maximum token length. Defaults to 512.
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get an item from the dataset by index.

        Args:
            idx (int): The index of the desired sample.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: A tuple containing tokenized code,
                                          docstring tensors, and the lang_token_id.
        """
        sample = self.data[idx]
        code = sample["code"]
        docstring = sample["docstring"]
        language = sample["language"]

        # Get the corresponding special token using the dictionary
        lang_token = self.tokenizer.lang_to_token[language.lower()]
        # Convert the lang_token to its corresponding ID
        lang_token_id = self.tokenizer.convert_tokens_to_ids(lang_token)
        lang_token_id_tensor = torch.tensor([lang_token_id], dtype=torch.long)

        tokenized_code = self.tokenizer(
            code,
            truncation=True,
            max_length=self.max_length + 1,
            padding="max_length",
            return_tensors="pt",
        )

        tokenized_docstring = self.tokenizer(
            docstring,
            truncation=True,
            max_length=self.max_length + 1,
            padding="max_length",
            return_tensors="pt",
        )

        return (
            tokenized_code["input_ids"].squeeze(0),
            tokenized_docstring["input_ids"].squeeze(0),
            lang_token_id_tensor,
        )


def download_dataset_from_kaggle(path="data"):
    """
    Download the CodeSearchNet dataset from Kaggle.
    Make sure to have the Kaggle API token in ~/.kaggle/kaggle.json

    Returns:
        str: Path to the downloaded dataset.
    """
    import kaggle

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(
        "omduggineni/codesearchnet", path=path, unzip=True
    )


def load_local_dataset(lang="all", path="data"):
    """
    Load a local dataset from the downloaded Kaggle dataset.

    Args:
        lang (str): The language to be used for the dataset.
        path (str, optional): Path to the downloaded dataset. Defaults to "data".

    Returns:
        Dataset: dataset loaded from local files
    """
    path = Path(path)

    if lang != "all":
        # Read the downloaded dataset
        path = path / lang / lang / "final/jsonl"
        dataset = load_dataset(
            "json",
            data_files={
                "train": glob.glob(path.as_posix() + "/train/*.jsonl"),
                "validation": glob.glob(path.as_posix() + "/valid/*.jsonl"),
                "test": glob.glob(path.as_posix() + "/test/*.jsonl"),
            },
        )
    else:
        train_files = glob.glob(path.as_posix() + "/**/train/*.jsonl", recursive=True)
        valid_files = glob.glob(path.as_posix() + "/**/valid/*.jsonl", recursive=True)
        test_files = glob.glob(path.as_posix() + "/**/test/*.jsonl", recursive=True)
        dataset = load_dataset(
            "json",
            data_files={
                "train": train_files,
                "validation": valid_files,
                "test": test_files,
            },
        )

    return dataset


def get_dataloader(dataset, tokenizer, args):
    """
    Get the data. Either train or validation.
    Filter out examples with more than 512 tokens (or args.max_function_length).

    Returns:
        DataLoader: Filtered training dataset in DataLoader.
    """
    # Filter out examples with more than 512 tokens (or args.max_function_length)
    dataset = dataset.filter(
        lambda example: filter_by_token_length(
            tokenizer, example, args.max_function_length
        )
    )

    dataset = CodeSearchNetDataset(
        dataset, tokenizer, max_length=args.max_function_length
    )

    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )


if __name__ == "__main__":
    download_dataset_from_kaggle()
