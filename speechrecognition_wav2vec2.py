import torch
import torchaudio
from torchaudio.utils import download_asset
from pydub import AudioSegment
from pydub.silence import split_on_silence
import os

def readytext(wavpath):

    # print(torch.__version__)
    # print(torchaudio.__version__)

    torch.random.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # print(device)

    # https://pytorch.org/audio/stable/generated/torchaudio.models.Wav2Vec2Model.html#torchaudio.models.Wav2Vec2Model
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_LARGE_LV60K_960H
    # bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    print("Sample Rate:", bundle.sample_rate)
    # print("Labels:", bundle.get_labels())
    model = bundle.get_model().to(device)
    print(model.__class__)

    class GreedyCTCDecoder(torch.nn.Module):
        def __init__(self, labels, blank=0):
            super().__init__()
            self.labels = labels
            self.blank = blank

        def forward(self, emission: torch.Tensor) -> str:
            """Given a sequence emission over labels, get the best path string
            Args:
              emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.

            Returns:
              str: The resulting transcript
            """
            indices = torch.argmax(emission, dim=-1)  # [num_seq,]
            indices = torch.unique_consecutive(indices, dim=-1)
            indices = [i for i in indices if i != self.blank]
            return "".join([self.labels[i] for i in indices])

    def recognizer(wavepath):
        waveform, sample_rate = torchaudio.load(wavepath)
        waveform = waveform.to(device)

        if sample_rate != bundle.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)

        with torch.inference_mode():
            features, _ = model.extract_features(waveform)

        with torch.inference_mode():
            emission, _ = model(waveform)

        decoder = GreedyCTCDecoder(labels=bundle.get_labels())
        transcript = decoder(emission[0])

        return transcript


    sound = AudioSegment.from_file(wavpath)

    chunks = split_on_silence(sound,
            # experiment with this value for your target audio file
            min_silence_len = 500,
            # adjust this per requirement
            silence_thresh = sound.dBFS-16,
            # keep the silence for 1 second, adjustable as well
            keep_silence=100,
        )

    whole_text = ''

    for i, audio_chunk in enumerate(chunks, start=1):
        # export audio chunk and save it in
        # the `folder_name` directory.
        chunk_filename = os.path.join(f"chunk{i}.wav")
        audio_chunk.export(chunk_filename, format="wav")

        # recognize the chunk
        text = recognizer(chunk_filename)
        print(i, ":", text)
        whole_text += text

        os.remove(chunk_filename)

    print("whole text : ", whole_text)

        # file = open("script1.txt", "w")
        # file.write(whole_text)
        # file.close()

    return whole_text