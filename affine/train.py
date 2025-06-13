from affine.model import AffineLightning, GenerationCallback
import torch
from torch.utils.data import DataLoader
import dac
from dac import DAC
from svae.model import PairedWaveformDataset
import argparse
import pytorch_lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger
from utils.checkpoints import get_checkpoint_path

# constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLE_RATE = 48000

def main(args):
    NUM_EPOCHS = args.epochs
    BATCH_SIZE = args.batchsize
    LR = args.lr
    TEMPO = args.bpm
    SUBDIVS = args.subdivs
    TEST_FILE = args.test
    TEST_OUT = args.out
    TEST_FREQ = args.outfreq
    CKPT_LOAD = args.loadckpt
    SORT_KEY = args.ckptkey
    DESCENDING = not args.asc
    EXPERIMENT_NAME = args.name
    LAMBDA_EMB = args.lemb
    LAMBDA_ADV = args.ladv
    WARMUP = args.warmup
    NFFT = 1024

    query_dir = args.query
    target_dir = args.target

    dac_model = DAC.load(dac.utils.download()).to(DEVICE)
    model_sr = dac_model.sample_rate
    block_length_in_samples = int(model_sr*60/(TEMPO*SUBDIVS/4))

    dataset = PairedWaveformDataset(query_dir, target_dir, block_length_in_samples, SAMPLE_RATE)
    train_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE)

    # the length of an audio block may be altered during decoding.
    # thus, a second block sample length must be passed to the discriminator
    with torch.inference_mode():
        dummy_frame = dac_model.encode(train_loader.dataset[0][0].unsqueeze(0))[0]
        block_length_in_frames = dummy_frame.shape[2]
        output_block = dac_model.decode(dummy_frame)
        output_block_length_in_samples = output_block.shape[2]
        dummy_stft = torch.stft(output_block.squeeze(1), NFFT, return_complex=True, window=torch.hann_window(NFFT, device=output_block.device)).abs()

    # gan = DACGAN(dac_model, device, block_length_in_samples, output_block_length_in_samples, block_length_in_frames, lambda_embedding=lambda_embedding) # initialize new model
    gan = AffineLightning(block_length_in_samples, output_block_length_in_samples, block_length_in_frames, [dummy_stft.shape[1], dummy_stft.shape[2]], NFFT, LAMBDA_EMB, LAMBDA_ADV, dac_model, warmup=WARMUP, lr = LR)

    tblogger = TensorBoardLogger(save_dir="affine_logs", name=EXPERIMENT_NAME)
    trainer = pl.Trainer(accelerator="auto", devices=1, max_epochs=NUM_EPOCHS, callbacks=GenerationCallback(block_length_in_samples, TEST_FILE ,TEST_OUT, TEST_FREQ), logger=tblogger)

    # load from previously saved checkpoint, if provided
    ckpt = get_checkpoint_path(CKPT_LOAD, SORT_KEY, DESCENDING) if CKPT_LOAD is not None else None

    trainer.fit(gan, train_dataloaders=train_loader, ckpt_path=ckpt)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description="Train single-instrument VAE model.")

    parser.add_argument("--query", help="Location of query data.", type=str, metavar="path", required=True)
    parser.add_argument("--target", help="Location of target data.", type=str, metavar="path", required=True)
    parser.add_argument("--epochs", help="Max number of training epochs", type=int, metavar="epochs", default=100)
    parser.add_argument("--batchsize", help="Batch size", type=int, metavar="size", default=32)
    parser.add_argument("--lr", help="Optimizer learning rate", type=float, metavar="rate", default=1e-4)
    parser.add_argument("--bpm", help="Model tempo", metavar="bpm", type=int, default=90)
    parser.add_argument("--subdivs", help="Segments per bar", metavar="divs", type=int, default=8)
    parser.add_argument("--test", help="Test model on audio file after training", type=str, metavar="audio_file_path", default=None)
    parser.add_argument("--out", help="Name of output test file dir", type=str, metavar="dirname", default=None)
    parser.add_argument("--outfreq", help="How often to generate test outputs (once every <epochs>)", type=int, metavar="epochs", default=5)
    parser.add_argument("--loadckpt", help="Resume training from checkpoint in lightning_logs folder.", type=str, metavar="checkpoint_folder_path", default=None)
    parser.add_argument("--ckptkey", help="Sorting key for checkpoint in folder.", type=str, metavar="key", default="step")
    parser.add_argument("--asc", help="Sort checkpoints according to key in ascending order.", action="store_true")
    parser.add_argument("--name", help="Name of experiment.", type=str, metavar="experiment_name", default="lightning_logs")
    parser.add_argument("--ladv", help="Set importance of adversarial loss in generator cost function.", type=float, metavar="lambda", default=1)
    parser.add_argument("--lemb", help="Set importance of feature distance loss in generator cost function.", type=float, metavar="lambda", default=1)
    parser.add_argument("--warmup", help="Number of epochs in warmup phase (no discriminator)", type=int, metavar="epochs", default=250)
    args=parser.parse_args()
    main(args)

