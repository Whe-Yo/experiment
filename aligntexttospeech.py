import torch
import torchaudio
import numpy as np
from dataclasses import dataclass

def readytextalign(wavpath, speechscript):

    # print(torch.__version__)
    # print(torchaudio.__version__)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(device)

    torch.random.manual_seed(0)


    SPEECH_FILE = wavpath
    # SPEECH_FILE = torchaudio.utils.download_asset("tutorial-assets/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav")

    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    model = bundle.get_model().to(device)
    labels = bundle.get_labels()
    with torch.inference_mode():
        waveform, _ = torchaudio.load(SPEECH_FILE)

        # 분할 작업을 수행할 배치 크기 및 스텝 크기 설정
        batch_size = 4  # 원하는 배치 크기로 조절하세요
        step_size = 8000  # 분할 스텝 크기로 조절하세요
        # 결과를 저장할 리스트 초기화
        all_emissions = []
        # waveform을 로드하고 GPU로 전송
        waveform = waveform.to(device)
        # 전체 waveform을 작은 배치로 나누고 이를 이용하여 처리
        total_samples = waveform.size(1)
        for start in range(0, total_samples, step_size):
            end = min(start + step_size, total_samples)
            batch_waveform = waveform[:, start:end]
            with torch.no_grad():
                emissions, _ = model(batch_waveform)
            emissions = torch.log_softmax(emissions, dim=-1)
            # 결과를 리스트에 추가
            all_emissions.append(emissions)

        # 모든 작은 배치의 결과를 연결하여 전체 결과를 얻음
        full_emissions = torch.cat(all_emissions, dim=1)

        emissions = torch.log_softmax(full_emissions, dim=-1)

    emission = emissions[0].cpu().detach()

    # print(labels)

    # We enclose the transcript with space tokens, which represent SOS and EOS.
    # f = open("data/script1.txt")
    # transcript = f.readline()
    transcript = speechscript
    # transcript = "|I|HAD|THAT|CURIOSITY|BESIDE|ME|AT|THIS|MOMENT|"
    dictionary = {c: i for i, c in enumerate(labels)}

    tokens = [dictionary[c] for c in transcript]
    # print(list(zip(transcript, tokens)))


    def get_trellis(emission, tokens, blank_id=0):
        num_frame = emission.size(0)
        num_tokens = len(tokens)

        trellis = torch.zeros((num_frame, num_tokens))
        trellis[1:, 0] = torch.cumsum(emission[1:, blank_id], 0)
        trellis[0, 1:] = -float("inf")
        trellis[-num_tokens + 1 :, 0] = float("inf")

        for t in range(num_frame - 1):
            trellis[t + 1, 1:] = torch.maximum(
                # Score for staying at the same token
                trellis[t, 1:] + emission[t, blank_id],
                # Score for changing to the next token
                trellis[t, :-1] + emission[t, tokens[1:]],
            )
        return trellis


    trellis = get_trellis(emission, tokens)

    @dataclass
    class Point:
        token_index: int
        time_index: int
        score: float


    def backtrack(trellis, emission, tokens, blank_id=0):
        t, j = trellis.size(0) - 1, trellis.size(1) - 1

        path = [Point(j, t, emission[t, blank_id].exp().item())]
        while j > 0:
            # Should not happen but just in case
            assert t > 0

            # 1. Figure out if the current position was stay or change
            # Frame-wise score of stay vs change
            p_stay = emission[t - 1, blank_id]
            p_change = emission[t - 1, tokens[j]]

            # Context-aware score for stay vs change
            stayed = trellis[t - 1, j] + p_stay
            changed = trellis[t - 1, j - 1] + p_change

            # Update position
            t -= 1
            if changed > stayed:
                j -= 1

            # Store the path with frame-wise probability.
            prob = (p_change if changed > stayed else p_stay).exp().item()
            path.append(Point(j, t, prob))

        # Now j == 0, which means, it reached the SoS.
        # Fill up the rest for the sake of visualization
        while t > 0:
            prob = emission[t - 1, blank_id].exp().item()
            path.append(Point(j, t - 1, prob))
            t -= 1

        return path[::-1]


    path = backtrack(trellis, emission, tokens)
    # for p in path:
    #     print(p)

    # Merge the labels
    @dataclass
    class Segment:
        label: str
        start: int
        end: int
        score: float

        def __repr__(self):
            return f"{self.label} {self.score:4.2f} {self.start} {self.end}"

        @property
        def length(self):
            return self.end - self.start


    def merge_repeats(path):
        i1, i2 = 0, 0
        segments = []
        while i1 < len(path):
            while i2 < len(path) and path[i1].token_index == path[i2].token_index:
                i2 += 1
            score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
            segments.append(
                Segment(
                    transcript[path[i1].token_index],
                    path[i1].time_index,
                    path[i2 - 1].time_index + 1,
                    score,
                )
            )
            i1 = i2
        return segments


    segments = merge_repeats(path)
    # for seg in segments:
    #     print(seg)

    # Merge words
    def merge_words(segments, separator="|"):
        words = []
        i1, i2 = 0, 0
        while i1 < len(segments):
            if i2 >= len(segments) or segments[i2].label == separator:
                if i1 != i2:
                    segs = segments[i1:i2]
                    word = "".join([seg.label for seg in segs])
                    score = sum(seg.score * seg.length for seg in segs) / sum(seg.length for seg in segs)
                    words.append(Segment(word, segments[i1].start, segments[i2 - 1].end, score))
                i1 = i2 + 1
                i2 = i1
            else:
                i2 += 1
        return words


    word_segments = merge_words(segments)

    # file = open("textalign1.txt", "w")
    for word in word_segments:
        print(word)
        # file.write(str(word))
        # file.write('\n')
    # file.close()

    return word_segments