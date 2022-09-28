import ffmpeg
import glob
import torch
import os
import re
import json
import sys
import functools
import logging
import yaml
import speechbrain
from multiprocessing import Pool, reduction
from speechbrain.lobes.features import Fbank
from speechbrain.lobes.models import ECAPA_TDNN
from speechbrain.lobes.models import Xvector
from speechbrain.pretrained import EncoderClassifier
from speechbrain.utils.checkpoints import Checkpointer
from speechbrain.lobes.augment import EnvCorrupt
from speechbrain.lobes.augment import TimeDomainSpecAugment
from speechbrain.dataio.dataset import DynamicItemDataset
from speechbrain.dataio.encoder import CategoricalEncoder
from os.path import exists
from pathlib import Path
from tqdm import tqdm
from pprint import pprint


logger = logging.getLogger(__name__)


class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x


class LanguageBrain(speechbrain.core.Brain):
    def on_stage_start(self, stage, epoch):
        if stage == speechbrain.Stage.TRAIN:
            for module in [self.modules.compute_features, self.modules.mean_var_norm, 
                           self.modules.embedding_model, self.modules.classifier]:
                for p in module.parameters():
                    p.requires_grad = True

        # enable grad for all modules we want to fine-tune
        if stage != speechbrain.Stage.TRAIN:
            self.error_metrics =  speechbrain.utils.metric_stats.MetricStats(metric=speechbrain.nnet.losses.classification_error)
    
    def compute_forward(self, batch, stage):
        """Computation pipeline based on a encoder + speaker classifier.
        Data augmentation and environmental corruption are applied to the
        input speech.
        """
        batch = batch.to(self.device)
        wavs, lens = batch.sig
        wavs, lens = wavs.to(self.device), lens.to(self.device)

        if stage == speechbrain.Stage.TRAIN:

            # Applying the augmentation pipeline
            wavs_aug_tot = []
            wavs_aug_tot.append(wavs)

            # Apply augment
            wavs_aug = self.hparams.augment_speed(wavs, lens)
            wavs_aug = self.hparams.add_rev_noise(wavs, lens)
            # Managing speed change
            if wavs_aug.shape[1] > wavs.shape[1]:
                wavs_aug = wavs_aug[:, 0 : wavs.shape[1]]
            else:
                zero_sig = torch.zeros_like(wavs)
                zero_sig[:, 0 : wavs_aug.shape[1]] = wavs_aug
                wavs_aug = zero_sig
           
            wavs = wavs_aug
            wavs_aug_tot[0] = wavs

            wavs = torch.cat(wavs_aug_tot, dim=0)
            self.n_augment = len(wavs_aug_tot)
            lens = torch.cat([lens] * self.n_augment)

        # Feature extraction and normalization
        feats = self.modules.compute_features(wavs)
        feats = self.modules.mean_var_norm(feats, lens)

        # Embeddings + speaker classifier
        embeddings = self.modules.embedding_model(feats, lens)
        outputs = self.modules.classifier(embeddings)

        return outputs, lens

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss using speaker-id as label.
        """
        predictions, lens = predictions
        uttid = batch.id
        langid = batch.lang_id_encoded

        # Concatenate labels (due to data augmentation)
        if stage == speechbrain.Stage.TRAIN:
            langid = torch.cat([langid] * self.n_augment, dim=0)

        # breakpoint()
        loss = self.hparams.compute_cost(predictions, langid.unsqueeze(1), lens)

        if hasattr(self.hparams.lr_annealing, "on_batch_end"):
            self.hparams.lr_annealing.on_batch_end(self.optimizer)

        if stage != speechbrain.Stage.TRAIN:
            self.error_metrics.append(
                uttid, predictions, langid.unsqueeze(1), lens, reduction="batch"
            )

        return loss

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of an epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == speechbrain.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["ErrorRate"] = self.error_metrics.summarize("average")

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == speechbrain.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(epoch)
            speechbrain.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            pprint({"epoch": epoch, "lr": old_lr})
            pprint(self.train_stats)
            pprint(stage_stats)
          
            self.checkpointer.save_and_keep_only(
                meta={"ErrorRate": stage_stats["ErrorRate"]},
                min_keys=["ErrorRate"],
            )



def convert_sample_rate(data_path, audio_path, new_audio):
    """Converts files into 16kHZ and wav
    """
    stream = ffmpeg.input(audio_path)
    audio = stream.audio
    stream = ffmpeg.output(audio, os.path.join(data_path, new_audio), **{'ar': '16000'})
    try:
        ffmpeg.run(stream, capture_stdout=True, capture_stderr=True, overwrite_output=True)
    except ffmpeg.Error as e:
        print(e.stderr)

def audio_preprocessing_pipeline(audio_path, n_workers):
    """Pipeline for audio files convertation in one folder with multiprocessing
    """
    folders = os.listdir(audio_path)
    noises = ("RIRS_NOISES", "rirs_noises.zip", "noise.csv", "reverb.csv")
    for folder in set(folders):
        if folder not in noises:
            pprint("Preprocessing " + folder) 
            data_path = os.path.join(audio_path, folder)
            f_process = functools.partial(audio_preprocessing, data_path=data_path)
            files = os.listdir(os.path.join(audio_path, folder))
            pool = Pool(n_workers)
            pool.map(f_process, tqdm(files))
            pool.close()
            pool.terminate()

def audio_preprocessing(file_, data_path):
    """Convertation of audio files in all folders
    """
    audio = os.path.join(data_path, file_)
    if exists(audio):
        name = file_
        name = name.split(".")
        if name[1] != "txt":
            if name[0].split("_")[-1] != "converted":
                name = name[0] + "_converted.wav"
                if not exists(os.path.join(data_path, name)):
                    convert_sample_rate(data_path, audio, name)
                    os.remove(audio)

def parse_to_json(output_file, audio_path, old_model, new_model):
    """Parse audio files into json format.
    Path should contain folder for each language with audio files and txt file with language name (e.g, "ky: Kyrgyz")
    """
    label_encoder = CategoricalEncoder()
    label_encoder.load_or_create(path=os.path.join(old_model, "label_encoder.txt"))
    old_encoder_len = len(label_encoder)
    examples = {}
    folders = set(os.listdir(audio_path))
    noises = ("RIRS_NOISES", "rirs_noises.zip", "noise.csv", "reverb.csv")
    for folder in folders:
        if folder not in noises:
            wav_files = glob.glob(os.path.join(audio_path, folder, "*.wav"),
                                recursive=True)
            meta_txt = glob.glob(os.path.join(audio_path, folder, "*.txt"),
                                recursive=True)
            with open(meta_txt[0]) as f:
                lang_name = f.readlines()
                lang_name = lang_name[0]
                label_encoder.ensure_label(lang_name)
        
            for idx, utterance in enumerate(wav_files):
                utt_id = Path(utterance).stem
                examples[utt_id] = {"audio.pth": utterance,
                                    "language_id": lang_name}

    with open(output_file, "w") as f:
        json.dump(examples, f, indent=4)
    
    # Save label_encoder (in case if languages in pretrain and finetune differ)
    label_encoder.save(os.path.join(new_model, "label_encoder.txt"))
    n_languages = len(label_encoder)

    return label_encoder, n_languages==old_encoder_len

def train(hparams, train_dataset, valid_dataset, run_opts, n_languages, n_languages_match):
    
    # Initialize ddp (useful only for multi-GPU DDP training)
    speechbrain.utils.distributed.ddp_init_group(run_opts)

    # Load pretrained model
    language_id = EncoderClassifier.from_hparams(source=hparams["old_model"], savedir="tmp")
    
    # Cut last layer of classifier and create new one if number of languages in pretrain and finetune differ
    if not n_languages_match:
        language_id.mods.classifier.out.w = Identity()
        language_id.mods.classifier.out.w = torch.nn.Linear(512, n_languages)

    modules = {
        "compute_features": language_id.mods.compute_features, # we use the same features 
        "mean_var_norm": language_id.mods.mean_var_norm,
        "embedding_model": language_id.mods.embedding_model, 
        "classifier": language_id.mods.classifier}
    
    # Cut and create new classifier layer again
    modules["classifier"].out.w = Identity()
    modules["classifier"].out.w = torch.nn.Linear(512, n_languages)
    
    checkpoint_dir = hparams["new_model"]
    checkpointer = Checkpointer(checkpoint_dir, 
                                recoverables = {"normalizer": modules['mean_var_norm'],
                                                "embedding_model": modules['embedding_model'], 
                                                "classifier": modules['classifier']})
    
    train_params = {
        "compute_cost": lambda x, y, z: speechbrain.nnet.losses.nll_loss(x, y, z),
        "lr_annealing": speechbrain.nnet.schedulers.LinearScheduler(hparams["lr"], 
                                                                    hparams["lr_final"],
                                                                    hparams["number_of_epochs"]),
        "n_languages": 108,
        "checkpointer": checkpointer,
        "n_languages_match": n_languages_match,
        "n_languages": n_languages,
        "augment_speed": TimeDomainSpecAugment(sample_rate=16000, speeds=[90, 100, 110]),
        "add_rev_noise": EnvCorrupt(
                                                        openrir_folder=hparams["data_path"], 
                                                        openrir_max_noise_len=1.0, 
                                                        reverb_prob=0.4,
                                                        noise_prob=0.5, 
                                                        noise_snr_low=0, 
                                                        noise_snr_high=11, 
                                                        rir_scale_factor=0.85
                                                        )
        }
    
    
    brain = LanguageBrain(modules, hparams=train_params, 
                          opt_class=lambda x: torch.optim.Adam(x, 1e-5), 
                          checkpointer=checkpointer,
                          run_opts=run_opts)
    
    print("Start training")
    brain.fit(range(hparams["number_of_epochs"]), train_dataset, valid_dataset,
                        {"batch_size": hparams["batch_size"], 
                        "drop_last":True, 
                        "shuffle": True},
                        {"batch_size": hparams["batch_size"], 
                        "drop_last":True, 
                        "shuffle": True})


if __name__ == "__main__":

    torch.backends.cudnn.benchmark = True

    hparams_file, run_opts, _ = speechbrain.parse_arguments(sys.argv[1:])

    with open(hparams_file, "r", encoding="utf-8") as stream:
        hparams = yaml.safe_load(stream)
    

    if hparams["audio_preprocessing"]:
        audio_preprocessing_pipeline(hparams["data_path"], hparams["n_workers"])
    
    if hparams["model_training"]:
        pprint("Start loading data")
        label_encoder, n_languages_match = parse_to_json("data.json", hparams["data_path"], hparams["old_model"], hparams["new_model"])
        dataset = DynamicItemDataset.from_json("data.json")
        dataset.add_dynamic_item(label_encoder.encode_label, takes="language_id", provides="lang_id_encoded")
        dataset.add_dynamic_item(speechbrain.dataio.dataio.read_audio, takes="audio.pth", provides="sig")
        dataset.set_output_keys(["id", "sig", "lang_id_encoded", "audio.pth"])
        _, _ = parse_to_json("data_val.json", hparams["data_val_path"], hparams["old_model"], hparams["new_model"])
        val_set = DynamicItemDataset.from_json("data_val.json")
        val_set.add_dynamic_item(label_encoder.encode_label, takes="language_id", provides="lang_id_encoded")
        val_set.add_dynamic_item(speechbrain.dataio.dataio.read_audio, takes="audio.pth", provides="sig")
        val_set.set_output_keys(["id", "sig", "lang_id_encoded", "audio.pth"])
        
        with open(os.path.join(hparams["old_model"], "hyperparams.yaml"), "r+") as f:
            old_hparams = f.read()
        
        n_languages = len(label_encoder)
    
        # Change hyperparameters according to new data
        new_hparams = re.sub("out_n_neurons: (.*)", "out_n_neurons: " + str(n_languages), old_hparams)
        new_hparams = re.sub("pretrained_path: (.*)", "pretrained_path: <insert path>", new_hparams)
    
        with open(os.path.join(hparams["new_model"], "hyperparams.yaml"), "w") as f:
            f.write(new_hparams)
    
        train(hparams, dataset, val_set, run_opts, n_languages, n_languages_match)

        os.remove("data.json")
        os.remove("data_val.json")

    print("End of running")
