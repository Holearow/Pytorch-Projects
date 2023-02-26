class Config():
    def __init__(
            self,
            device=None,
            batch_size=64,
            epochs=10,
            learning_rate=1e-4,
            max_source_len=40,
            max_target_len=80,
            pad_token_id=0,
            unk_token_id=1,
            sos_token_id=2,
            eos_token_id=3,
            input_dim=0,
            output_dim=0,
            enc_emb_dim=256,
            dec_emb_dim=256,
            enc_hid_dim=512,
            dec_hid_dim=512,
            enc_dropout=0.1,
            dec_dropout=0.1
    ):

        # training setting & hyper parameters
        self.device = device
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len
        # special tokens
        self.pad_token_id = pad_token_id
        self.unk_token_id = unk_token_id
        self.sos_token_id = sos_token_id
        self.eos_token_id = eos_token_id
        # model parameters
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.enc_emb_dim = enc_emb_dim
        self.dec_emb_dim = dec_emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.enc_dropout = enc_dropout
        self.dec_dropout = dec_dropout
