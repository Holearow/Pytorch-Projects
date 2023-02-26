import os
import torch
import torch.nn as nn
import torch.optim as optim
import math
import time
import argparse
from torch.utils.data import DataLoader
from dataloader import RestaurantDataset
from config import Config
from nltk.tokenize import word_tokenize
from torch.nn.utils.rnn import pad_sequence
from model import Encoder, Decoder, Attention, Seq2Seq

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def transfer_to_ids(t, mode, default_value):
    # src
    if mode == 0:
        ids = [src_vocab.get(token, default_value) for token in t]
    # trg
    else:
        ids = [trg_vocab.get(token, default_value) for token in t]

    ids = torch.tensor(ids)
    return ids


def convert_to_features(batch):
    # 1.get batch and tokenize
    table_batch = [s[0] for s in batch]
    reference_batch = [s[1] for s in batch]

    # 2.tokenize, then add <eos> & <sos> tokens
    table_tokens = [['<sos>'] + word_tokenize(t) + ['<eos>'] for t in table_batch]
    reference_tokens = [['<sos>'] + word_tokenize(t) + ['<eos>'] for t in reference_batch]

    # 3.order by the length of table
    sorted_idx = sorted(range(len(table_tokens)), key=lambda x: len(table_tokens[x]), reverse=True)
    ordered_table_tokens = [table_tokens[i] for i in sorted_idx]
    ordered_reference_tokens = [reference_tokens[i] for i in sorted_idx]

    # 4.get table_length & reference_length (type:tensor)
    table_length = torch.tensor([len(t) for t in ordered_table_tokens])
    reference_length = torch.tensor([len(t) for t in ordered_reference_tokens])

    # 5.convert to id sequence (type: tensor)
    table_ids = [transfer_to_ids(t, 0, cfg.unk_token_id) for t in ordered_table_tokens]
    reference_ids = [transfer_to_ids(t, 1, cfg.unk_token_id) for t in ordered_reference_tokens]

    # 6.padding
    table_tensor = pad_sequence(table_ids, batch_first=True, padding_value=cfg.pad_token_id)
    reference_tensor = pad_sequence(reference_ids, batch_first=True, padding_value=cfg.pad_token_id)

    # 7.form batch tensors
    encodings = {
        'input_ids': table_tensor.transpose(1, 0).contiguous(),
        'input_len': table_length,
        'decoder_input_ids': reference_tensor.transpose(1, 0).contiguous(),
        'decoder_input_len': reference_length
    }

    return encodings


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def train(model, dataloader, optimizer, criterion):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(dataloader):
        optimizer.zero_grad()
        # 1.get output
        # src = [src_seq_len, bsz], src_len = [bsz], trg = [trg_seq_len, bsz]
        src = batch['input_ids'].to(cfg.device)
        src_len = batch['input_len'].to(cfg.device)
        trg = batch['decoder_input_ids'].to(cfg.device)

        # output = [trg_seq_len-1, bsz, output_dim]
        output = model(
            src=src,
            src_len=src_len,
            trg=trg,
            teacher_forcing_ratio=0.0
        )

        # 2.calculate the loss
        output_dim = output.shape[-1]

        output = output.view(-1, output_dim)
        trg = trg[1:].view(-1)

        # trg = [(trg_seq_len - 1) * bsz]
        # output = [(trg_seq_len - 1) * bsz, output_dim]
        loss = criterion(output, trg)

        # 3. loss backward
        loss.backward()
        optimizer.step()

        # 4. accumulate the loss
        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)


def evaluate(model, dataloader, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            # 1.get output
            # src = [src_seq_len, bsz], src_len = [bsz], trg = [trg_seq_len, bsz]
            src = batch['input_ids'].to(cfg.device)
            src_len = batch['input_len'].to(cfg.device)
            trg = batch['decoder_input_ids'].to(cfg.device)

            # output = [trg_seq_len-1, bsz, output_dim]
            output = model(
                src=src,
                src_len=src_len,
                trg=trg,
                teacher_forcing_ratio=0     # disable teacher forcing
            )

            # 2.calculate the loss
            output_dim = output.shape[-1]
            output = output.view(-1, output_dim)
            trg = trg[1:].view(-1)

            # trg = [(trg_seq_len - 1) * bsz]
            # output = [(trg_seq_len - 1) * bsz, output_dim]
            loss = criterion(output, trg)

            # 3. accumulate the loss
            epoch_loss += loss.item()

    return epoch_loss / len(dataloader)


def main():
    # access to data
    train_set = RestaurantDataset(mode='train')
    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=convert_to_features
    )
    valid_set = RestaurantDataset(mode='valid')
    valid_loader = DataLoader(
        valid_set,
        batch_size=cfg.batch_size,
        collate_fn=convert_to_features
    )

    # model initialization
    attention = Attention(
        enc_hid_dim=cfg.enc_hid_dim,
        dec_hid_dim=cfg.dec_hid_dim
    )
    encoder = Encoder(
        input_dim=cfg.input_dim,
        emb_dim=cfg.enc_emb_dim,
        enc_hid_dim=cfg.enc_hid_dim,
        dec_hid_dim=cfg.dec_hid_dim,
        dropout=cfg.enc_dropout
    )
    decoder = Decoder(
        output_dim=cfg.output_dim,
        emb_dim=cfg.dec_emb_dim,
        enc_hid_dim=cfg.enc_hid_dim,
        dec_hid_dim=cfg.dec_hid_dim,
        dropout=cfg.dec_dropout,
        attention=attention
        )
    model = Seq2Seq(
        encoder=encoder,
        decoder=decoder,
        src_pad_idx=cfg.pad_token_id,
        device=cfg.device
    ).to(device)

    # define optimizer & criterion
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=cfg.pad_token_id)

    # train process
    best_valid_loss = float('inf')
    best_train_loss = float('inf')
    for epoch in range(cfg.epochs):
        # 1.train
        start_time = time.time()
        train_loss = train(model, train_loader, optimizer, criterion)

        # 2.validate
        valid_loss = evaluate(model, valid_loader, criterion)
        end_time = time.time()

        # 3.calculate the time
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        # 4.save best checkpoint (metric: valid_loss)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), f'./checkpoint/best_valid_checkpoint.pt')

        if train_loss < best_train_loss:
            best_train_loss = train_loss
            torch.save(model.state_dict(), f'./checkpoint/best_train_checkpoint.pt')

        # 5.report
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

    print('-------------------------------')
    print(f'Best valid loss:{best_valid_loss:.3f}')
    print(f'Best train loss:{best_train_loss:.3f}')
    print('-------------------------------')


def transfer_to_sentence(t):
    tokens = []
    for idx in t:
        if idx == cfg.sos_token_id or idx == cfg.pad_token_id or idx == cfg.unk_token_id:
            continue
        elif idx == cfg.eos_token_id:
            break
        else:
            token = trg_vocab_reverse[idx]
            tokens.append(token)

    sentence = ' '.join(tokens)
    return sentence


def predict():
    # access to data
    test_set = RestaurantDataset(mode='test')
    test_loader = DataLoader(
        test_set,
        batch_size=cfg.batch_size,
        collate_fn=convert_to_features
    )

    # model initialization
    attention = Attention(cfg.enc_hid_dim, cfg.dec_hid_dim)
    encoder = Encoder(cfg.input_dim, cfg.enc_emb_dim, cfg.enc_hid_dim, cfg.dec_hid_dim, cfg.enc_dropout)
    decoder = Decoder(cfg.output_dim, cfg.dec_emb_dim, cfg.enc_hid_dim, cfg.dec_hid_dim, cfg.dec_dropout, attention)
    model = Seq2Seq(encoder, decoder, cfg.pad_token_id, cfg.device).to(device)

    # load checkpoint & predict
    model.load_state_dict(torch.load('checkpoint/best_train_checkpoint.pt'))
    model.eval()

    predict_reference = []

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            # src = [src_seq_len, bsz], src_len = [bsz]
            src = batch['input_ids'].to(cfg.device)
            src_len = batch['input_len'].to(cfg.device)

            # encoder_outputs = [src_len, bsz, enc_hid_dim * 2]
            # hidden = [bsz, dec_hid_dim]
            encoder_outputs, hidden = model.encoder(src, src_len)

            # mask = [bsz, src_len]
            mask = model.create_mask(src)

            # first input to the decoder is the <sos> tokens
            bsz = src.shape[1]
            input = torch.zeros(bsz).int().to(cfg.device)
            input[:] = cfg.sos_token_id

            # tensor to store decoder outputs
            predictions = torch.zeros(cfg.max_target_len, bsz).to(cfg.device)
            # outputs = torch.zeros(cfg.max_target_len, cfg.batch_size, cfg.output_dim).to(cfg.device)
            # decoding
            for t in range(cfg.max_target_len):
                output, hidden, _ = model.decoder(input, hidden, encoder_outputs, mask)
                top1 = output.argmax(1)
                predictions[t] = top1
                input = top1

            predictions = predictions.transpose(1, 0).contiguous()
            pred_batch = [transfer_to_sentence(t.tolist()) for t in predictions]
            predict_reference.extend(pred_batch)

    # save
    with open('predictions.txt', 'w', encoding='utf-8') as f:
        for reference in predict_reference:
            f.write(reference + '\n')


if __name__ == "__main__":
    # parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train')
    args = parser.parse_args()

    # get the vocabulary
    global src_vocab
    global trg_vocab
    global trg_vocab_reverse

    src_vocab = dict()
    with open('vocab/source_vocab.tsv', 'r', encoding='utf-8') as f1:
        for line in f1:
            index, token = line.split('\t')
            token = token.strip('\n')
            src_vocab[token] = int(index)

    trg_vocab = dict()
    trg_vocab_reverse = dict()
    with open('vocab/target_vocab.tsv', 'r', encoding='utf-8') as f2:
        for line in f2:
            index, token = line.split('\t')
            token = token.strip('\n')
            trg_vocab[token] = int(index)
            trg_vocab_reverse[int(index)] = token

    # define arguments
    global cfg
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = Config(
        device=device,
        batch_size=128,
        epochs=5,
        learning_rate=1e-5,
        max_source_len=40,
        max_target_len=80,
        input_dim=len(src_vocab),
        output_dim=len(trg_vocab),
        enc_emb_dim=32,
        dec_emb_dim=256,
        enc_hid_dim=64,
        dec_hid_dim=512,
        enc_dropout=0.0,
        dec_dropout=0.2
    )

    # train/infer
    if args.mode == 'train':
        main()
    else:
        predict()
