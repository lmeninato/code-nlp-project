import argparse
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import one_hot

from data import get_dataloader, load_local_dataset
from utils import (
    generate_square_subsequent_mask,
    create_padding_mask,
    get_code_tokenizer,
    get_english_tokenizer,
    save_model,
)
from model import TransformerEncoderModel, TransformerDecoderModel


def validate_model(
    encoder,
    decoder_docstring,
    decoder_code,
    optimizer,
    criterion,
    valid_dataloader,
    validation_losses,
    epoch,
    device,
    code_tokenizer_pad_idx,
    english_input_dim,
    code_input_dim,
):
    encoder.eval()
    decoder_docstring.eval()
    decoder_code.eval()

    progress_bar = tqdm(valid_dataloader, desc=f"Validation Epoch {epoch + 1}")
    with torch.no_grad():
        for i, (code_input, docstring_input, lang_token_id) in enumerate(progress_bar):
            # Get the code and docstring tensors
            # Move the input tensors to the device
            code_input = code_input.to(device).transpose(0, 1)  # (seq_len, batch_size)
            docstring_input = docstring_input.to(device).transpose(
                0, 1
            )  # (seq_len, batch_size)
            lang_token_id = lang_token_id.to(device)  # (batch_size, 1)

            # # Generate masks

            code_padding_mask = create_padding_mask(
                code_input, code_tokenizer_pad_idx
            )  # (batch_size, seq_len)

            code_mask = generate_square_subsequent_mask(code_input.size(0)).to(
                device
            )  # (sq_len+1, sq_len+1)
            docstring_mask = generate_square_subsequent_mask(
                docstring_input.size(0)
            ).to(
                device
            )  # (sq_len, sq_len)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass through the encoder and decoders
            code_representation = encoder(
                code_input, code_padding_mask
            )  # (seq_len, batch_size, d_model)
            reconstructed_docstring = decoder_docstring(
                docstring_input, code_representation, docstring_mask
            )  # (seq_len, batch_size, vocab_size)
            reconstructed_code = decoder_code(
                code_input, code_representation, code_mask, lang_token_id
            )  # (seq_len+1, batch_size, vocab_size)

            # Calculate the loss

            loss_docstring = criterion(
                reconstructed_docstring,
                one_hot(docstring_input, num_classes=english_input_dim).float(),
            )
            loss_code = criterion(
                reconstructed_code,
                one_hot(code_input, num_classes=code_input_dim).float(),
            )
            loss = loss_docstring + loss_code

            # Backward pass and optimization
            loss.backward()

            # Update loss statistics
            if i % 100 == 99:
                tqdm.write(f"Batch: {i + 1}, Validation Loss: {loss.item()}")

            # Update the tqdm progress bar with the loss
            progress_bar.set_postfix(Loss=loss.item())
            validation_losses.append(loss.item())


def train_model(
    dataloader,
    valid_dataloader,
    device,
    code_tokenizer_pad_idx,
    english_input_dim,
    code_input_dim,
    num_epochs=5,
    d_model=256,
    d_hid=512,
    num_layers=2,
    nhead=4,
    dropout=0.1,
):
    encoder = TransformerEncoderModel(
        code_input_dim, d_model, nhead, d_hid, num_layers, dropout
    ).to(device)
    decoder_docstring = TransformerDecoderModel(
        english_input_dim, d_model, nhead, d_hid, num_layers, dropout
    ).to(device)
    decoder_code = TransformerDecoderModel(
        code_input_dim, d_model, nhead, d_hid, num_layers, dropout
    ).to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    params = (
        list(encoder.parameters())
        + list(decoder_docstring.parameters())
        + list(decoder_code.parameters())
    )
    optimizer = optim.Adam(params, lr=0.001)
    train_losses = []
    validation_losses = []

    for epoch in range(num_epochs):
        encoder.train()
        decoder_docstring.train()
        decoder_code.train()

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")
        for i, (code_input, docstring_input, lang_token_id) in enumerate(progress_bar):
            # Get the code and docstring tensors
            # Move the input tensors to the device
            code_input = code_input.to(device).transpose(0, 1)  # (seq_len, batch_size)
            docstring_input = docstring_input.to(device).transpose(
                0, 1
            )  # (seq_len, batch_size)
            lang_token_id = lang_token_id.to(device)  # (batch_size, 1)

            # # Generate masks

            code_padding_mask = create_padding_mask(
                code_input, code_tokenizer_pad_idx
            )  # (batch_size, seq_len)

            code_mask = generate_square_subsequent_mask(code_input.size(0)).to(
                device
            )  # (sq_len+1, sq_len+1)
            docstring_mask = generate_square_subsequent_mask(
                docstring_input.size(0)
            ).to(
                device
            )  # (sq_len, sq_len)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass through the encoder and decoders
            code_representation = encoder(
                code_input, code_padding_mask
            )  # (seq_len, batch_size, d_model)
            reconstructed_docstring = decoder_docstring(
                docstring_input, code_representation, docstring_mask
            )  # (seq_len, batch_size, vocab_size)
            reconstructed_code = decoder_code(
                code_input, code_representation, code_mask, lang_token_id
            )  # (seq_len+1, batch_size, vocab_size)

            # Calculate the loss

            loss_docstring = criterion(
                reconstructed_docstring,
                one_hot(docstring_input, num_classes=english_input_dim).float(),
            )
            loss_code = criterion(
                reconstructed_code,
                one_hot(code_input, num_classes=code_input_dim).float(),
            )
            loss = loss_docstring + loss_code

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Update loss statistics
            if i % 100 == 99:
                tqdm.write(f"Batch: {i + 1}, Loss: {loss.item()}")

            # Update the tqdm progress bar with the loss
            progress_bar.set_postfix(Loss=loss.item())
            train_losses.append(loss.item())

        validate_model(
            encoder=encoder,
            decoder_docstring=decoder_docstring,
            decoder_code=decoder_code,
            valid_dataloader=valid_dataloader,
            device=device,
            code_tokenizer_pad_idx=code_tokenizer_pad_idx,
            english_input_dim=english_input_dim,
            code_input_dim=code_input_dim,
            criterion=criterion,
            validation_losses=validation_losses,
            epoch=epoch,
        )

    encoder, decoder_docstring, decoder_code, train_losses, validation_losses


def main():
    parser = argparse.ArgumentParser(
        description="Trains model on CodeSearchNet dataset"
    )

    parser.add_argument(
        "--max_function_length", type=int, default=512, help="Maximum function length"
    )
    parser.add_argument(
        "--d_model", type=int, default=256, help="Dimension of the model"
    )
    parser.add_argument(
        "--d_hid", type=int, default=512, help="Dimension of the hidden layer"
    )
    parser.add_argument("--num_layers", type=int, default=2, help="Number of layers")
    parser.add_argument(
        "--nhead",
        type=int,
        default=4,
        help="Number of heads in the multi-head attention mechanism",
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers")
    parser.add_argument("--data_dir", type=str, default="data", help="Data dir")
    parser.add_argument("--output_dir", type=str, default="models", help="Output dir")

    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    english_language_tokenizer = get_english_tokenizer()
    code_tokenizer = get_code_tokenizer()
    # dataset = load_dataset("code_search_net", "python")
    dataset = load_local_dataset("python", args.data_dir)

    train_dataloader = get_dataloader(
        dataset["train"], code_tokenizer, english_language_tokenizer, args
    )
    valid_dataloader = get_dataloader(
        dataset["validation"], code_tokenizer, english_language_tokenizer, args
    )

    (
        encoder_model,
        decoder_docstring_model,
        decoder_code_model,
        train_losses,
        valid_losses,
    ) = train_model(
        dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        device=device,
        code_tokenizer_pad_idx=code_tokenizer.pad_token_id,
        english_input_dim=len(english_language_tokenizer),
        code_input_dim=len(code_tokenizer),
        d_model=args.d_model,
        d_hid=args.d_hid,
        num_layers=args.num_layers,
        nhead=args.nhead,
        dropout=args.dropout,
    )

    # TODO output this stuff to a file
    print(f"Got training losses: {train_losses}")
    print(f"Got validation losses: {valid_losses}")

    save_model(encoder_model, f"{args.output_dir}/encoder_model.pt")
    save_model(decoder_docstring_model, f"{args.output_dir}/decoder_docstring_model.pt")
    save_model(decoder_code_model, f"{args.output_dir}/decoder_code_model.pt")


if __name__ == "__main__":
    main()
