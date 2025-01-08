import torch
import torch.nn
import torchaudio
import streamlit as st
import torchaudio.pipelines

class CTCDecoder(nn.Module):
    def __init__(self, uploaded_file, blank=0):
        super().__init__()
        self.uploaded_file = uploaded_file
        self.blank = blank
        self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
        self.taudio = torchaudio
        self.bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
        self.model = self.bundle.get_model().to(self.device)
        self.labels = self.bundle.get_labels()
        self.waveform, self.sample_rate = self.taudio.load(self.uploaded_file)
        self.waveform = self.waveform.to(self.device)
        if self.sample_rate != self.bundle.sample_rate:
            self.waveform = self.taudio.functional.resample(self.waveform, self.sample_rate, self.bundle.sample_rate)
        with torch.inference_mode():
            self.features, _ = self.model.extract_features(self.waveform)
            self.final_emission, _ = self.model(self.waveform)

    def forward(self):
        idx = torch.argmax(self.final_emission[0], dim=-1)
        idx = torch.unique_consecutive(idx, dim=-1)
        idx = [i for i in idx if i != self.blank]
        return "".join([self.labels[i] for i in idx])

def main():
    st.title("Speech to text")
    uploaded_file = None
    if uploaded_file is None:
        uploaded_file = st.file_uploader("Upload audio file", type = 'wav')
        if uploaded_file is not None:
            st.write("Audio file uploaded")
            if st.button("Process..."):
                decoder = CTCDecoder(uploaded_file)
                transcript = decoder()
                if transcript is not None:
                    st.write(f"Transcript: {transcript}")
                else:
                    st.write("Error generating transcript")

if __name__ == '__main__':
    main()
