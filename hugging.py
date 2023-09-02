# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("audio-classification", model="CAiRE/SER-wav2vec2-large-xlsr-53-eng-zho-adults")
# Load model directly
from transformers import AutoProcessor, Wav2Vec2ForMultilabelSequenceClassification

processor = AutoProcessor.from_pretrained("CAiRE/SER-wav2vec2-large-xlsr-53-eng-zho-adults")
model = Wav2Vec2ForMultilabelSequenceClassification.from_pretrained("CAiRE/SER-wav2vec2-large-xlsr-53-eng-zho-adults")
