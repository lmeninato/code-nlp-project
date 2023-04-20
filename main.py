import argparse
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim

from data import get_dataloader, load_local_dataset
from utils import generate_square_subsequent_mask, get_tokenizer, save_model
from model import TransformerEncoderModel, TransformerDecoderModel


def train_model(
    dataloader,
    valid_dataloader,
    device,
    input_dim,
    num_epochs=10,
    d_model=256,
    d_hid=512,
    num_layers=2,
    nhead=4,
    dropout=0.1,
):
    encoder = TransformerEncoderModel(
        input_dim, d_model, nhead, d_hid, num_layers, dropout
    ).to(device)
    decoder_docstring = TransformerDecoderModel(
        input_dim, d_model, nhead, d_hid, num_layers, dropout
    ).to(device)
    decoder_code = TransformerDecoderModel(
        input_dim, d_model, nhead, d_hid, num_layers, dropout
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
            code_input = code_input.to(device)
            docstring_input = docstring_input.to(device)
            lang_token_id = lang_token_id.to(device)

            # Generate masks
            code_mask = generate_square_subsequent_mask(code_input.size(0)).to(device)
            docstring_mask = generate_square_subsequent_mask(
                docstring_input.size(0)
            ).to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass through the encoder and decoders
            code_representation = encoder(code_input, code_mask)
            reconstructed_docstring = decoder_docstring(
                docstring_input, code_representation, docstring_mask
            )
            reconstructed_code = decoder_code(
                code_input, code_representation, code_mask, lang_token_id
            )
            # Calculate the loss
            loss_docstring = criterion(
                reconstructed_docstring.permute(1, 2, 0),
                docstring_input.transpose(0, 1),
            )
            loss_code = criterion(
                reconstructed_code.permute(1, 2, 0), code_input.transpose(0, 1)
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

        encoder.eval()
        decoder_docstring.eval()
        decoder_code.eval()

        progress_bar = tqdm(valid_dataloader, desc=f"Validation Epoch {epoch + 1}")
        with torch.no_grad():
            for i, (code_input, docstring_input, lang_token_id) in enumerate(
                progress_bar
            ):
                # Get the code and docstring tensors
                # Move the input tensors to the device
                code_input = code_input.to(device)
                docstring_input = docstring_input.to(device)
                lang_token_id = lang_token_id.to(device)

                # Generate masks
                code_mask = generate_square_subsequent_mask(code_input.size(0)).to(
                    device
                )
                docstring_mask = generate_square_subsequent_mask(
                    docstring_input.size(0)
                ).to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass through the encoder and decoders
                code_representation = encoder(code_input, code_mask)
                reconstructed_docstring = decoder_docstring(
                    docstring_input, code_representation, docstring_mask
                )
                reconstructed_code = decoder_code(
                    code_input, code_representation, code_mask, lang_token_id
                )

                # Calculate the loss
                loss_docstring = criterion(
                    reconstructed_docstring.permute(1, 2, 0),
                    docstring_input.transpose(0, 1),
                )
                loss_code = criterion(
                    reconstructed_code.permute(1, 2, 0), code_input.transpose(0, 1)
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

    tokenizer = get_tokenizer()
    # dataset = load_dataset("code_search_net", "python")
    dataset = load_local_dataset("python", args.data_dir)

    train_dataloader = get_dataloader(dataset["train"], tokenizer, args)
    valid_dataloader = get_dataloader(dataset["validation"], tokenizer, args)

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
        input_dim=len(tokenizer),
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
