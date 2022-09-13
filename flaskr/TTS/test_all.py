import os, librosa, scipy, uuid, argparse
import tensorflow as tf
import numpy as np
import soundfile as sf
from models.tacotron import Tacotron, post_CBHG
from models.modules import griffin_lim
from util.hparams import *
from util.text import text_to_sequence
from pinyin.parse_text_to_pyin import get_pyin
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from pathlib import Path

suffix = str(Path(os.path.abspath(__file__)).parent)

checkpoint1_dir = os.path.join(suffix, 'checkpoint/1')
checkpoint2_dir = os.path.join(suffix, 'checkpoint/2')


model1 = Tacotron(K=16, conv_dim=[128, 128])
checkpoint1 = tf.train.Checkpoint(model=model1)
checkpoint1.restore(tf.train.latest_checkpoint(checkpoint1_dir)).expect_partial()

model2 = post_CBHG(K=8, conv_dim=[256, mel_dim])
optimizer = Adam()
step = tf.Variable(0)
checkpoint2 = tf.train.Checkpoint(optimizer=optimizer, model=model2, step=step)
checkpoint2.restore(tf.train.latest_checkpoint(checkpoint2_dir)).expect_partial()

mel_targets = pad_sequences([np.load(os.path.join(suffix, 'speaker/mel-00001.npy'))], padding='post', dtype='float32')


def test_step1(text, mel):
    pyin, _ = get_pyin(text)
    seq = text_to_sequence(pyin)
    enc_input = np.asarray([seq], dtype=np.int32)
    sequence_length = np.asarray([len(seq)], dtype=np.int32)
    dec_input = np.zeros((1, sequence_length[0]*2, mel_dim), dtype=np.float32)
    enc_output = None

    pred = []
    for i in range(1, sequence_length[0]*2+1):
        # print('in', enc_input, sequence_length, dec_input)
        mel_out, alignment, enc_output = model1(enc_input, sequence_length, dec_input, is_training=False, mel_targets=mel, enc_output=enc_output)
        # print('out', mel_out, alignment)
        if i < sequence_length[0]*2:
            dec_input[:, i, :] = mel_out[:, reduction * i - 1, :]
        pred.extend(mel_out[:, reduction * (i-1) : reduction * i, :])

    pred = np.reshape(np.asarray(pred), [-1, mel_dim])
    # alignment = np.squeeze(alignment, axis=0)

    return pred

def test_step2(mel):
    mel = np.expand_dims(mel, axis=0)
    pred = model2(mel, is_training=False)

    pred = np.squeeze(pred, axis=0)
    pred = np.transpose(pred)

    pred = (np.clip(pred, 0, 1) * max_db) - max_db + ref_db
    pred = np.power(10.0, pred * 0.05)
    wav = griffin_lim(pred ** 1.5)
    wav = scipy.signal.lfilter([1], [1, -preemphasis], wav)
    wav = librosa.effects.trim(wav, frame_length=win_length, hop_length=hop_length)[0]
    wav = wav.astype(np.float32)
    return wav
    
def production(text):
    return test_step2(test_step1(text, mel_targets))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", help="輸出的資料夾")
    parser.add_argument("--mode", required=True, choices=["test", "save"])
    parser.add_argument("--text", default='缺少輸入', help="要轉成語音的文字")

    args = parser.parse_args()

    def main():
        if args.mode == 'test':
            if args.output_dir is None:
                parser.error("--mode test 需要 --output_dir。")
            sf.write(os.path.join(args.output_dir, '{}.wav'.format(uuid.uuid1())), production(args.text), sample_rate)
        elif args.mode == 'save':
            model1.save_weights('./model_save_data/tacotron', save_format="tf")
            model2.save_weights('./model_save_data/post_CBHG', save_format="tf")
    main()