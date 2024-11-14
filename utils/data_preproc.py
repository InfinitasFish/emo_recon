from sklearn.preprocessing import LabelEncoder
import os
import numpy as np
from PIL import Image
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
import librosa
import librosa.display
from pathlib import Path
import logging

import torch
from pytorchvideo.data.clip_sampling import ConstantClipsPerVideoSampler
from pytorchvideo.data.encoded_video import EncodedVideo
from torchvision import transforms
import torchaudio


def get_ds_info(csv_path):
    dev_csv_path = '../MELD.Raw/dev_sent_emo.csv'
    df = pd.read_csv(dev_csv_path)
    print(df.columns)

    row_ex = df.iloc[0]
    print(row_ex)

    plt.figure(figsize=(12,6))
    plt.hist(df['Emotion'], bins='auto')
    plt.show()

    emo_rates = {}
    for ue in df['Emotion'].unique():
        emo_rates[ue] = len(df[df['Emotion'] == ue]) / len(df['Emotion'])

    for k, v in emo_rates.items():
        print(f'{k} rate: {v:.4f}')


def get_pil_frames(video_path, num_frames=15):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0

    while frame_count < num_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame))
        frame_count += 1

    cap.release()

    if len(frames) < num_frames:
        raise ValueError(f"Video contains fewer than {num_frames} frames.")

    return frames


def get_wav_from_mp4(video_path, out_dir='../MELD.Raw/dev/dev_wave/'):
    out_fname = Path(video_path).stem
    output_path = Path(out_dir + f'{out_fname}.wav')
    if output_path.exists():
        return output_path

    # if not exists convert mp4 to wav
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(output_path, codec='pcm_s16le', fps=16000)
    return output_path


def create_dataset_split(csv_path, videos_path):
    df = pd.read_csv(csv_path)
    label_encoder = LabelEncoder()
    # not used
    df['Emotion_encoded'] = label_encoder.fit_transform(df['Emotion'])

    X, y = [], []
    video_files = [file for file in os.listdir(videos_path)
                  if file.endswith('.mp4') and not file.startswith('.')]

    for video_file in video_files:
        v_path = os.path.join(videos_path, video_file)
        try:
            id_info = video_file.split('.mp4')[0].split('_')
            if (len(id_info) != 2
                or not id_info[0][3:].isdigit()
                or not id_info[1][3:].isdigit()):

                raise ValueError(f'invalid video fname format {video_file}')

            diag_id, utt_id = int(id_info[0][3:]), int(id_info[1][3:])
            utt_row = df[(df['Dialogue_ID'] == diag_id) & (df['Utterance_ID'] == utt_id)]

            if not utt_row.empty:
                true_label = utt_row['Emotion'].values[0]
                video_path = v_path
                audio_data_path = get_wav_from_mp4(v_path)
                text_data = utt_row['Utterance'].values[0]

                X.append([text_data, video_path, audio_data_path])
                y.append([true_label])

        except:
            print(f'Invalid data for video {v_path}')

    return X, y


# TODO: adapt imagebind code
def load_transform_audio_data(
    audio_paths,
    device,
    num_mel_bins=128,
    target_length=204,
    sample_rate=16000,
    clip_duration=2,
    clips_per_video=3,
    mean=-4.268,
    std=9.138,
):
    if audio_paths is None:
        return None

    audio_outputs = []
    clip_sampler = ConstantClipsPerVideoSampler(
        clip_duration=clip_duration, clips_per_video=clips_per_video
    )

    for audio_path in audio_paths:
        waveform, sr = torchaudio.load(audio_path)
        if sample_rate != sr:
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=sr, new_freq=sample_rate
            )
        all_clips_timepoints = get_clip_timepoints(
            clip_sampler, waveform.size(1) / sample_rate
        )
        all_clips = []
        for clip_timepoints in all_clips_timepoints:
            waveform_clip = waveform[
                :,
                int(clip_timepoints[0] * sample_rate) : int(
                    clip_timepoints[1] * sample_rate
                ),
            ]
            waveform_melspec = waveform2melspec(
                waveform_clip, sample_rate, num_mel_bins, target_length
            )
            all_clips.append(waveform_melspec)

        normalize = transforms.Normalize(mean=mean, std=std)
        all_clips = [normalize(ac).to(device) for ac in all_clips]

        all_clips = torch.stack(all_clips, dim=0)
        audio_outputs.append(all_clips)

    return torch.stack(audio_outputs, dim=0)

# use it instead my get_pil_frames
def get_clip_timepoints(clip_sampler, duration):
    # Read out all clips in this video
    all_clips_timepoints = []
    is_last_clip = False
    end = 0.0
    while not is_last_clip:
        start, end, _, _, is_last_clip = clip_sampler(end, duration, annotation=None)
        all_clips_timepoints.append((start, end))
    return all_clips_timepoints


DEFAULT_AUDIO_FRAME_SHIFT_MS = 10
def waveform2melspec(waveform, sample_rate, num_mel_bins, target_length):
    # Based on https://github.com/YuanGongND/ast/blob/d7d8b4b8e06cdaeb6c843cdb38794c1c7692234c/src/dataloader.py#L102
    waveform -= waveform.mean()
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform,
        htk_compat=True,
        sample_frequency=sample_rate,
        use_energy=False,
        window_type="hanning",
        num_mel_bins=num_mel_bins,
        dither=0.0,
        frame_length=25,
        frame_shift=DEFAULT_AUDIO_FRAME_SHIFT_MS,
    )
    # Convert to [mel_bins, num_frames] shape
    fbank = fbank.transpose(0, 1)
    # Pad to target_length
    n_frames = fbank.size(1)
    p = target_length - n_frames
    # if p is too large (say >20%), flash a warning
    if abs(p) / n_frames > 0.2:
        logging.warning(
            "Large gap between audio n_frames(%d) and "
            "target_length (%d). Is the audio_target_length "
            "setting correct?",
            n_frames,
            target_length,
        )
    # cut and pad
    if p > 0:
        fbank = torch.nn.functional.pad(fbank, (0, p), mode="constant", value=0)
    elif p < 0:
        fbank = fbank[:, 0:target_length]
    # Convert to [1, mel_bins, num_frames] shape, essentially like a 1
    # channel image
    fbank = fbank.unsqueeze(0)
    return fbank


def main():
    videos_path = '../MELD.Raw/dev/dev_splits_complete/'
    csv_path = '../MELD.Raw/dev_sent_emo.csv'
    # X, y = create_dataset_split(csv_path, videos_path)
    #
    # print(len(X), len(y))
    # for i in range(5):
    #     print(X[i][0], X[i][1], X[i][2], y[i][0])

    return 0


if __name__ == '__main__':
    #dev_csv_path = '../MELD.Raw/dev_sent_emo.csv'
    #get_ds_info(dev_csv_path)
    main()

