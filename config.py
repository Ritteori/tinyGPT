EMBED_DIM = 512
NUM_HEADS = 8
MAX_SEQ_LEN = 256
DEPTH = 3
HIDDEN_DIM_MULTIPLICATOR = 4
DROPOUT = 0.2

BATCH_SIZE = 16
LR = 3e-4
EPOCHS = 12

DATA_DIR = r'data'
LOG_DIR = r'logs'
CHECKPOINTS_DIR = r'checkpoints'

TRAIN_ENCODED_TEXTES = r'saved_tokenizer/train_encoded_textes.pth'
TEST_ENCODED_TEXTES = r'saved_tokenizer/test_encoded_textes.pth'
VOCAB_PATH = r'saved_tokenizer/vocab.pt'

MERGES = 3000