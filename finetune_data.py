from torch.utils.data import DataLoader, Dataset


class CodeSearchNetDatasetCustom(Dataset):
    """
    CodeSearchNet Dataset.

    This class provides a PyTorch Dataset implementation for the CodeSearchNet dataset.
    It tokenizes and processes the code and docstring samples to be used in a model.
    """

    def __init__(
        self,
        data,
        tokenizer,
        max_code_length=512,
        max_docstring_length=512,
    ):
        """
        Initialize the dataset with data, tokenizer, and max_length.

        Args:
            data (List[Dict]): A list of dictionaries containing code and docstring samples. # noqa
            tokenizer (PreTrainedTokenizer): The tokenizer to be used for tokenization.
            max_length (int, optional): The maximum token length. Defaults to 512.
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_code_length = max_code_length
        self.max_docstring_length = max_docstring_length

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

        tokenized_code = self.tokenizer(
            code,
            truncation=True,
            max_length=self.max_code_length,
            padding="max_length",
            return_tensors="pt",
        )["input_ids"].squeeze(0)

        tokenized_docstring = self.tokenizer(
            docstring,
            truncation=True,
            max_length=self.max_docstring_length,
            padding="max_length",
            return_tensors="pt",
        )["input_ids"].squeeze(0)

        return (
            tokenized_code,
            tokenized_docstring,
        )


def custom_dataloader(
    dataset,
    tokenizer,
    max_function_length,
    max_docstring_length,
    batch_size,
    num_workers,
):
    def filter_by_token_length(example):
        tokenized_code = tokenizer(example["code"], return_length=True)
        tokenized_docstring = tokenizer(example["docstring"], return_length=True)

        def less_than_max_len(x, ml):
            return x["length"] <= ml

        return less_than_max_len(
            tokenized_code, max_function_length
        ) and less_than_max_len(tokenized_docstring, max_docstring_length)

    dataset = dataset.filter(filter_by_token_length)

    dataset = CodeSearchNetDatasetCustom(
        dataset,
        tokenizer,
        max_code_length=max_function_length,
        max_docstring_length=max_docstring_length,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
