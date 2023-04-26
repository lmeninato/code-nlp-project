import argparse
import os
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim

from data import get_dataloader, load_local_dataset, get_padded_tokenized_tensor
from utils import (
    generate_square_subsequent_mask,
    create_padding_mask,
    get_code_tokenizer,
    get_english_tokenizer,
    save_model,
    load_model,
)
from model import (
    TransformerEncoderModel,
    TransformerDecoderModel,
    GreedyDocstringDecoder,
    cross_entropy_loss_ignoring_padding,
)


def validate_model(
    encoder,
    decoder_docstring,
    decoder_code,
    criterion,
    valid_dataloader,
    validation_losses,
    epoch,
    device,
    code_tokenizer_pad_idx,
    code_coef,
    doc_coef,
    batches_per_epoch,
    decode_strategy_class,
    code_tokenizer,
    english_tokenizer,
    max_tgt_length,
):
    """Evaluates model on the validation set

    Args: See parameters to the train function
    """
    encoder.eval()
    decoder_docstring.eval()
    decoder_code.eval()

    progress_bar = tqdm(valid_dataloader, desc=f"Validation Epoch {epoch + 1}")
    with torch.no_grad():
        progress_bar = tqdm(range(batches_per_epoch))
        dataloader_iter = iter(valid_dataloader)
        loss = 0
        for i in progress_bar:
            code_input, docstring_input, lang_token_id = next(dataloader_iter)
            # Move the input tensors to the device
            code_input = code_input.to(device).transpose(0, 1)  # (seq_len, batch_size)
            docstring_input = docstring_input.to(device).transpose(
                0, 1
            )  # (seq_len, batch_size)
            # lang_token_id = lang_token_id.to(device)  # (batch_size, 1)

            # # Generate masks

            code_padding_mask = create_padding_mask(
                code_input, code_tokenizer_pad_idx
            )  # (batch_size, seq_len)

            # code_mask = generate_square_subsequent_mask(code_input.size(0)).to(
            #     device
            # )  # (sq_len+1, sq_len+1)
            docstring_mask = generate_square_subsequent_mask(
                docstring_input.size(0)
            ).to(
                device
            )  # (sq_len, sq_len)

            # Forward pass through the encoder and decoders
            code_representation = encoder(
                code_input, code_padding_mask
            )  # (batch_size, seq_len, d_model)
            reconstructed_docstring = decoder_docstring(
                docstring_input, code_representation, docstring_mask
            )  # (batch_size, seq_len, vocab_size)
            # reconstructed_code = decoder_code(
            #     code_input, code_representation, code_mask, lang_token_id
            # )  # (batch_size, seq_len+1, vocab_size)

            # Calculate the loss
            doc_output = reconstructed_docstring.permute(1, 0, 2).reshape(
                -1, reconstructed_docstring.shape[-1]
            )
            doc_target = docstring_input.transpose(0, 1).view(-1)
            loss_docstring = criterion(
                doc_output, doc_target, english_tokenizer.pad_token_id
            )

            # code_output = reconstructed_code.permute(1, 0, 2).reshape(
            #     -1, reconstructed_code.shape[-1]
            # )
            # code_target = code_input.transpose(0, 1).view(-1)
            # loss_code = criterion(code_output, code_target)

            # loss = doc_coef * loss_docstring + code_coef * loss_code
            loss = loss_docstring

            # Update loss statistics
            if i % batches_per_epoch == batches_per_epoch - 1:
                tqdm.write(f"Batch: {i + 1}, Validation Loss: {loss.item()}")
                decode_strategy = decode_strategy_class(encoder, decoder_docstring)
                decoded_code, decoded_docstring = decode_strategy.generate(
                    code_tokenizer,
                    english_tokenizer,
                    code_input[:, 0:1],
                    code_padding_mask[0:1, :],
                    english_tokenizer.bos_token_id,
                    english_tokenizer.eos_token_id,
                    max_tgt_length,
                )
                print("--------------- CODE INPUT ---------------")
                print(decoded_code)
                print("--------------- DOCSTRING OUTPUT ---------------")
                print(decoded_docstring)

            # Update the tqdm progress bar with the loss
            progress_bar.set_postfix(Loss=loss.item())
        validation_losses.append(loss.item())


def load_latest_model_checkpoint(to_update, checkpt_dir):
    """Load the state of the latest model checkpt.

    Args:
        checkpt_dir (str): The directory that contains model checkpoints to continue from

    Returns:
        int: the last epoch whose checkpoint was saved
    """
    latest_checkpt = sorted(os.listdir(checkpt_dir), reverse=True)[0]
    latest_checkpt_path = os.path.join(checkpt_dir, latest_checkpt)
    latest_checkpt_state = torch.load(latest_checkpt_path)
    to_update.load_state_dict(latest_checkpt_state)
    epoch = int(latest_checkpt.split(".")[0][len("epoch") :])
    return epoch


def epoch_model_save(enc, dec_code, dec_doc, output_dir, epoch):
    """Save all models from a given epoch.

    Args:
        epoch (int): The last epoch that was saved previously
    """
    checkpt_name = "epoch" + str(epoch).zfill(3)
    save_model(enc, f"{output_dir}/encoder_model/{checkpt_name}.pt")
    save_model(dec_doc, f"{output_dir}/decoder_docstring_model/{checkpt_name}.pt")
    save_model(dec_code, f"{output_dir}/decoder_code_model/{checkpt_name}.pt")


def get_memusage(obj: torch.Tensor):
    """Returns the memory usage of a tensor in MB."""
    return obj.element_size() * obj.nelement() / 1024 / 1024


def train_model(
    dataloader,
    valid_dataloader,
    device,
    code_tokenizer_pad_idx,
    english_input_dim,
    code_input_dim,
    output_dir,
    decode_strategy_class,
    code_tokenizer,
    english_tokenizer,
    max_tgt_length,
    num_epochs=5,
    d_model=256,
    d_hid=512,
    num_layers=2,
    nhead=4,
    dropout=0.1,
    epochs_per_save=10,
    batches_per_epoch=100,
    code_coef=0.5,
    doc_coef=0.5,
):
    """Trains the model.

    Args:
        dataloader (Dataloader): The dataloader containing the training data
        valid_dataloader (Dataloader): The dataloader containing the validation data
        device (str): Cpu or gpu
        code_tokenizer_pad_idx (int): The padding index
        english_input_dim (int): The number of english vocab words
        code_input_dim (int): The number of code vocab words
        output_dir (str): The output directory to store model checkpoints to
        num_epochs (int, optional): The number of training epochs to run. Defaults to 5.
        d_model (int, optional): Transformer model dimension. Defaults to 256.
        d_hid (int, optional): Transformer hidden dimension. Defaults to 512.
        num_layers (int, optional): The number of transformer layers for the encoder and decoder. Defaults to 2.
        nhead (int, optional): The number of attention heads to use. Defaults to 4.
        dropout (float, optional): The dropout fraction to use. Defaults to 0.1.
        epochs_per_save (int, optional): The number of epochs that go by before saving a checkpoint. Defaults to 10.
        batches_per_epoch (int, optional): The number of batches per epoch. Defaults to 100.
        code_coef (float, optional): A multiplier on the code recon loss. Defaults to 0.5.
        doc_coef (float, optional): A multiplier on the docstring recon loss. Defaults to 0.5.

    Returns:
        _type_: _description_
    """

    starting_epoch = 0

    encoder = TransformerEncoderModel(
        code_input_dim, d_model, nhead, d_hid, num_layers, dropout
    ).to(device)
    decoder_docstring = TransformerDecoderModel(
        english_input_dim, d_model, nhead, d_hid, num_layers, dropout
    ).to(device)
    decoder_code = TransformerDecoderModel(
        code_input_dim, d_model, nhead, d_hid, num_layers, dropout
    ).to(device)

    # if the checkpoint directory is not empty, start where the last run left off
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if os.listdir(output_dir):
        starting_epoch = load_latest_model_checkpoint(
            encoder, f"{output_dir}/encoder_model"
        )
        _ = load_latest_model_checkpoint(
            decoder_code, f"{output_dir}/decoder_code_model"
        )
        _ = load_latest_model_checkpoint(
            decoder_docstring, f"{output_dir}/decoder_docstring_model"
        )
        starting_epoch += 1
        print(f"starting from checkpoint. first epoch will be {starting_epoch}")

    # Define the loss function and optimizer
    criterion = cross_entropy_loss_ignoring_padding
    params = (
        list(encoder.parameters())
        + list(decoder_docstring.parameters())
        + list(decoder_code.parameters())
    )
    optimizer = optim.Adam(params, lr=0.001)
    train_losses = []
    validation_losses = []

    for epoch in range(starting_epoch, num_epochs):
        encoder.train()
        decoder_docstring.train()
        decoder_code.train()

        progress_bar = tqdm(range(batches_per_epoch))
        dataloader_iter = iter(dataloader)

        loss = 0
        for i in progress_bar:
            code_input, docstring_input, lang_token_id = next(dataloader_iter)
            # Get the code and docstring tensors
            # Move the input tensors to the device
            code_input = code_input.to(device).transpose(0, 1)  # (batch_size, seq_len)
            docstring_input = docstring_input.to(device).transpose(
                0, 1
            )  # (batch_size, seq_len)
            # lang_token_id = lang_token_id.to(device)  # (batch_size, 1)
            # # Generate masks
            code_padding_mask = create_padding_mask(
                code_input, code_tokenizer_pad_idx
            )  # (batch_size, seq_len)
            # code_mask = generate_square_subsequent_mask(code_input.size(0)).to(
            #     device
            # )  # (sq_len+1, sq_len+1)
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
            )  # (batch_size, seq_len, d_model)
            reconstructed_docstring = decoder_docstring(
                docstring_input, code_representation, docstring_mask
            )  # (batch_size, seq_len, vocab_size)
            # reconstructed_code = decoder_code(
            #     code_input, code_representation, code_mask, lang_token_id
            # )  # (batch_size, seq_len+1, vocab_size)

            # Calculate the loss
            doc_output = reconstructed_docstring.permute(1, 0, 2).reshape(
                -1, reconstructed_docstring.shape[-1]
            )
            doc_target = docstring_input.transpose(0, 1).view(-1)

            loss_docstring = criterion(
                doc_output, doc_target, english_tokenizer.pad_token_id
            )

            # code_output = reconstructed_code.permute(1, 0, 2).reshape(
            #     -1, reconstructed_code.shape[-1]
            # )
            # code_target = code_input.transpose(0, 1).view(-1)
            # loss_code = criterion(code_output, code_target)

            # loss = doc_coef * loss_docstring + code_coef * loss_code
            loss = loss_docstring

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
            criterion=criterion,
            validation_losses=validation_losses,
            epoch=epoch,
            code_coef=code_coef,
            doc_coef=doc_coef,
            batches_per_epoch=batches_per_epoch,
            decode_strategy_class=decode_strategy_class,
            code_tokenizer=code_tokenizer,
            english_tokenizer=english_tokenizer,
            max_tgt_length=max_tgt_length,
        )
        if epoch % epochs_per_save == 0:
            epoch_model_save(
                encoder, decoder_code, decoder_docstring, output_dir, epoch
            )

    epoch_model_save(encoder, decoder_code, decoder_docstring, output_dir, num_epochs)

    return train_losses, validation_losses


def parse_args():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser(
        description="Trains model on CodeSearchNet dataset"
    )

    parser.add_argument("--train", action="store_true")
    parser.add_argument("--inference", dest="train", action="store_false")
    parser.set_defaults(train=True)

    parser.add_argument(
        "--num_epochs", default=5, type=int, help="The number of epochs to train for"
    )
    parser.add_argument(
        "--max_function_length", type=int, default=512, help="Maximum function length"
    )
    parser.add_argument(
        "--max_docstring_length", type=int, default=512, help="Maximum docstring length"
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
    parser.add_argument(
        "--epochs_per_save", type=int, default=10, help="number of epochs between saves"
    )
    parser.add_argument(
        "--batches_per_epoch",
        type=int,
        default=100,
        help="the number of batches per epoch",
    )
    parser.add_argument(
        "--code_coefficient",
        type=float,
        default=0.5,
        help="multiplier for the code reconstruction loss",
    )
    parser.add_argument(
        "--doc_coefficient",
        type=float,
        default=0.5,
        help="multiplier for the doc reconstruction loss",
    )

    return parser.parse_args()


def main(args, device):
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

    train_losses, valid_losses = train_model(
        dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        device=device,
        code_tokenizer_pad_idx=code_tokenizer.pad_token_id,
        english_input_dim=len(english_language_tokenizer),
        code_input_dim=len(code_tokenizer),
        d_model=args.d_model,
        d_hid=args.d_hid,
        num_epochs=args.num_epochs,
        num_layers=args.num_layers,
        nhead=args.nhead,
        dropout=args.dropout,
        output_dir=args.output_dir,
        batches_per_epoch=args.batches_per_epoch,
        decode_strategy_class=GreedyDocstringDecoder,
        code_tokenizer=code_tokenizer,
        english_tokenizer=english_language_tokenizer,
        max_tgt_length=args.max_docstring_length,
    )

    # TODO output this stuff to a file
    print(f"Got training losses: {train_losses}")
    print(f"Got validation losses: {valid_losses}")


def get_models_for_inference(args, device):
    english_language_tokenizer = get_english_tokenizer()
    code_tokenizer = get_code_tokenizer()

    code_input_dim = len(code_tokenizer)
    english_input_dim = len(english_language_tokenizer)
    d_model = args.d_model
    d_hid = args.d_hid
    num_layers = args.num_layers
    nhead = args.nhead
    dropout = args.dropout

    encoder = TransformerEncoderModel(
        code_input_dim, d_model, nhead, d_hid, num_layers, dropout
    ).to(device)
    decoder_docstring = TransformerDecoderModel(
        english_input_dim, d_model, nhead, d_hid, num_layers, dropout
    ).to(device)
    decoder_code = TransformerDecoderModel(
        code_input_dim, d_model, nhead, d_hid, num_layers, dropout
    ).to(device)

    encoder = load_model(encoder, "models/encoder_model/epoch040.pt", device)
    decoder_docstring = load_model(
        decoder_docstring, "models/decoder_docstring_model/epoch040.pt", device
    )
    decoder_code = load_model(
        decoder_code, "models/decoder_code_model/epoch040.pt", device
    )

    dataset = load_local_dataset("python", args.data_dir)

    valid_dataloader = get_dataloader(
        dataset["validation"], code_tokenizer, english_language_tokenizer, args
    )

    return (
        encoder,
        decoder_docstring,
        decoder_code,
        code_tokenizer,
        english_language_tokenizer,
        valid_dataloader,
    )


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if args.train:
        main(args, device)
    else:
        (
            encoder,
            decoder_docstring,
            decoder_code,
            code_tok,
            eng_tok,
            valid_dataloader,
        ) = get_models_for_inference(args, device)
        raise Exception("TODO: run sample output - just use validation set for now")
