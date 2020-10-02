#!/usr/bin/env python3

from fastai.vision.all import *
from fastaudio.core.all import *
from fastaudio.augment.all import *
from efficientnet_pytorch import EfficientNet
from fastaudio.augment.spectrogram import CropTime

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def CrossValidationSplitter(col='fold', fold=1):
    "Split `items` (supposed to be a dataframe) by fold in `col`"
    def _inner(o):
        assert isinstance(o, pd.DataFrame), "ColSplitter only works when your items are a pandas DataFrame"
        col_values = o.iloc[:,col] if isinstance(col, int) else o[col]
        valid_idx = (col_values == fold).values.astype('bool')
        return IndexSplitter(mask2idxs(valid_idx))(o)
    return _inner


path = Path("/home/kkalyan/agri-split/data")
df = pd.read_csv(path/"Train_v.csv")
MODEL_SAVE = "/home/kkalyan/agri-split/scripts/models" 

accuracy = []

def run(fold):
    print("preparing fold {}".format(i))

    ct = CropTime(duration=1000)
    cfg = AudioConfig.BasicMelSpectrogram(n_fft=512)
    a2s = AudioToSpec.from_cfg(cfg)

    auds = DataBlock(blocks=(AudioBlock, CategoryBlock), 
                 get_x=ColReader("fn", pref=path), 
                 batch_tfms = [a2s],
                 splitter=CrossValidationSplitter(fold=i),
                 item_tfms=[ResizeSignal(3000), SignalShifter(), AddNoise(), ChangeVolume()],
                 get_y=ColReader("label"))
    dbunch = auds.dataloaders(df, bs=256)

    learn = cnn_learner(dbunch, 
                resnet18, 
                config=cnn_config(n_in=1),
                loss_fn=LabelSmoothingCrossEntropy,
                metrics=[accuracy]).to_fp16()
    
    learn.fine_tune(10)

    learn.fit_one_cycle(120, 1e-3, cbs=[EarlyStoppingCallback(patience=30), SaveModelCallback(fname=MODEL_SAVE)])
    learn.load(MODEL_SAVE)

    learn.unfreeze()
    learn.fit_one_cycle(100, slice(1e-5, 1e-3), cbs=[EarlyStoppingCallback(patience=30), SaveModelCallback(fname=MODEL_SAVE)])
    learn.load(MODEL_SAVE)

    learn.fit_one_cycle(100, slice(4e-7, 1e-5), cbs=[EarlyStoppingCallback(patience=30), SaveModelCallback(fname=MODEL_SAVE)])
    learn.load(MODEL_SAVE)
    
    learn.fit_one_cycle(100, slice(1e-8, 1e-6), cbs=[EarlyStoppingCallback(patience=30), SaveModelCallback(fname=MODEL_SAVE)])
    learn.load(MODEL_SAVE)

    preds, y = learn.get_preds(dl=dbunch.valid)
    acc = accuracy(preds, y)
    accuracy.append(acc)

if __name__ == "__main__":
    seed_everything(1234)
    for i in range(3):
        run(i+1)

    print("Accuracy: {}".format(np.mean(accuracy)))