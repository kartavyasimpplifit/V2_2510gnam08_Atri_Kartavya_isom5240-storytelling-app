# V2_2510gnam08_Atri_Kartavya_isom5240-storytelling-app
# Children's Story Generator

**ISOM5240 Assignment - Storytelling Application**


## Description

This is a storytelling application that generates stories from images using Hugging Face models.
The story is also converted to audio for children to listen.

## Models Used

1. Image to Text: nlpconnect/vit-gpt2-image-captioning
2. Story Generation: distilgpt2  
3. Text to Speech: facebook/mms-tts-eng

## Requirements

- Story must be at least 50 words
- Target audience is children aged 3-10 years
- Uses Hugging Face transformers pipelines

## How to Run

```bash
streamlit run app.py
