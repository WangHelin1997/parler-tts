import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer, set_seed
import soundfile as sf

device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = ParlerTTSForConditionalGeneration.from_pretrained("/export/fs05/hwang258/parler-tts/output_dir_training").to(device)
tokenizer = AutoTokenizer.from_pretrained("/export/fs05/hwang258/parler-tts/output_dir_training")

prompt = "Hey, how are you doing today? I like it."
description = "The woman delivers a slightly expressive and animated speech with a slow speed and low pitch."

input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

set_seed(43)

generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
audio_arr = generation.cpu().numpy().squeeze()
sf.write("parler_tts_out2.wav", audio_arr, model.config.sampling_rate)
