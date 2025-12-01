
ISOM5240 ASSIGNMENT - CHILDREN'S STORY GENERATOR

Student Name: Kartavya Atri
Date: November 2025

WHAT THIS DOES


This app takes an image and creates a children's story with audio narration.

How it works:
1. Upload an image
2. AI describes the image
3. AI writes a story (50+ words)
4. AI reads the story out loud

Target audience: Kids aged 3-10 years

MODELS USED

All models from Hugging Face:

1. Image to Text (350 MB)
   Model: nlpconnect/vit-gpt2-image-captioning
   Link: https://huggingface.co/nlpconnect/vit-gpt2-image-captioning
   Purpose: Describe what's in the image

2. Story Generation (548 MB)
   Model: gpt2
   Link: https://huggingface.co/gpt2
   Purpose: Write creative stories for children
   Note: I tried distilgpt2 first but stories were bad, so I used gpt2

3. Text to Speech (100 MB)
   Model: facebook/mms-tts-eng
   Link: https://huggingface.co/facebook/mms-tts-eng
   Purpose: Convert story to audio

Total Size: About 1 GB

FILES INCLUDED

app.py - Main application code
requirements.txt - Python packages needed
README.txt - This file

HOW TO RUN
=
Local:
1. pip install -r requirements.txt
2. streamlit run app.py
3. Open browser to http://localhost:8501

Online:
Visit: YOUR_STREAMLIT_URL_HERE

Note: First time loading takes 5-10 minutes (downloading models)


REQUIREMENTS MET
=

✓ Uses Hugging Face transformers pipelines
✓ Generates 50+ word stories
✓ Provides audio output
✓ Image upload functionality
✓ Target audience: children 3-10 years
==
KEY DECISIONS
==

Why gpt2 instead of distilgpt2?
- distilgpt2 is smaller (319 MB) but stories were repetitive and boring
- gpt2 is bigger (548 MB) but produces much better quality stories
- The extra 229 MB is worth it for better results

Story Generation Parameters:
- temperature: 0.85 (controls creativity)
- repetition_penalty: 1.5 (reduces repetitive text)
- max_new_tokens: 130 (story length control)

TESTING
=

Tested with:
- Animal images (cats, dogs, birds)
- Nature images (trees, flowers, landscapes)
- Object images (toys, vehicles, furniture)

All tests produced:
✓ Stories with 50+ words
✓ Child-friendly content
✓ Clear audio output


KNOWN ISSUES
=
- First load is slow (model downloads)
- Complex images get simplified descriptions
- Sometimes stories can be slightly repetitive

WHAT I LEARNED
==
- Prompt engineering really matters for AI quality
- Bigger models usually mean better results
- Parameter tuning is important for text generation
- Streamlit makes deployment easy

Reach me on kartavya.atri@u.nus.edu

================================================================================
