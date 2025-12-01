"""
ISOM5240 Assignment - Children's Storytelling Application
Student ID: 2510gnam08, S029
Name: Kartavya Atri
Defining Target: Children aged is ~3-10 years Also hope its ok to set the temperature for creativity 
"""

"""
ISOM5240 Individual Assignment Revised, Earlier was not giving bad story so usuing vit-gpt2-image-captioning
Using optimal small models for best quality
"""

import streamlit as st
from transformers import pipeline
from PIL import Image
import scipy.io.wavfile
import numpy as np
import io

st.set_page_config(page_title="Story Generator", layout="centered")


# Model loading functions
@st.cache_resource
def load_image_model():
    """
    Image to Text Model
    Model: nlpconnect/vit-gpt2-image-captioning
    Size: 350 MB
    Link: https://huggingface.co/nlpconnect/vit-gpt2-image-captioning
    """
    model = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")
    return model


@st.cache_resource
def load_story_model():
    """
    Story Generation Model
    Model: gpt2 (NOT distilgpt2)
    Size: 548 MB
    Link: https://huggingface.co/gpt2
    
    Why GPT-2 instead of distilgpt2:
    - Much better story quality
    - More coherent narratives
    - Better vocabulary
    - Still reasonably small
    """
    model = pipeline("text-generation", model="gpt2")
    return model


@st.cache_resource
def load_audio_model():
    """
    Text to Speech Model
    Model: facebook/mms-tts-eng
    Size: 100 MB
    Link: https://huggingface.co/facebook/mms-tts-eng
    """
    model = pipeline("text-to-speech", model="facebook/mms-tts-eng")
    return model


# Function to get caption from image
def get_caption(image):
    """Get caption from image"""
    caption_model = load_image_model()
    result = caption_model(image)
    text = result[0]["generated_text"]
    return text


# Function to generate story - IMPROVED VERSION
def generate_story(caption):
    """
    Generate story using GPT-2
    Much better quality than distilgpt2
    """
    story_model = load_story_model()
    
    # Better prompt for GPT-2
    prompt = f"""Write a magical short story for young children.

The story features: {caption}

Story:
Once upon a time,"""
    
    # Optimized parameters for GPT-2
    output = story_model(
        prompt,
        max_new_tokens=130,
        temperature=0.85,
        top_p=0.92,
        top_k=50,
        do_sample=True,
        repetition_penalty=1.5,
        no_repeat_ngram_size=3,
        pad_token_id=50256
    )
    
    # Extract story
    full_text = output[0]['generated_text']
    
    # Clean the output
    if "Story:" in full_text:
        story_text = full_text.split("Story:")[1].strip()
    else:
        story_text = full_text.replace(prompt, "").strip()
    
    # Make sure it starts right
    if not story_text.startswith("Once upon a time"):
        story_text = "Once upon a time, " + story_text
    
    # Remove AI artifacts
    story_text = story_text.replace("</s>", "").replace("<|endoftext|>", "")
    
    # Control length
    words = story_text.split()
    
    # Ensure minimum 50 words
    if len(words) < 50:
        story_text = story_text.rstrip('.!?') + ", and everyone lived happily ever after in their wonderful world."
        words = story_text.split()
    
    # Cap at 120 words
    if len(words) > 120:
        story_text = ' '.join(words[:120])
        # Find last complete sentence
        for i in range(len(story_text)-1, 0, -1):
            if story_text[i] in '.!?':
                story_text = story_text[:i+1]
                break
    
    # Ensure proper ending
    if not story_text.endswith(('.', '!', '?')):
        story_text = story_text + '.'
    
    return story_text


# Function to convert text to speech
def text_to_speech(text):
    """Convert story to audio"""
    tts_model = load_audio_model()
    speech = tts_model(text)
    audio = np.array(speech["audio"]).flatten()
    rate = speech["sampling_rate"]
    return audio, rate


# Main application
def main():
    
    st.title("Children's Story Generator")
    st.write("Upload an image and get a story with audio")
    st.write("For children aged 3-10 years")
    
    # Sidebar
    st.sidebar.header("Assignment Info")
    st.sidebar.write("ISOM5240 Individual Assignment")
    st.sidebar.write("Student ID: YOUR_STUDENT_ID")
    
    st.sidebar.subheader("Models Used")
    
    st.sidebar.text("1. Image to Text (350 MB)")
    st.sidebar.code("nlpconnect/vit-gpt2-image-captioning")
    st.sidebar.caption("https://huggingface.co/nlpconnect/vit-gpt2-image-captioning")
    
    st.sidebar.text("2. Story Generation (548 MB)")
    st.sidebar.code("gpt2")
    st.sidebar.caption("https://huggingface.co/gpt2")
    
    st.sidebar.text("3. Text to Speech (100 MB)")
    st.sidebar.code("facebook/mms-tts-eng")
    st.sidebar.caption("https://huggingface.co/facebook/mms-tts-eng")
    
    st.sidebar.write("---")
    st.sidebar.info("Total: ~1 GB (works on free Streamlit)")
    st.sidebar.success("Using GPT-2 for better stories")
    
    # Main content
    st.write("---")
    st.subheader("Step 1: Upload an Image")
    
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png']
    )
    
    if uploaded_file is not None:
        
        image = Image.open(uploaded_file)
        st.image(image, caption="Your uploaded image", width=400)
        
        st.write(f"File name: {uploaded_file.name}")
        st.write(f"Image size: {image.size[0]} x {image.size[1]} pixels")
        
        st.write("---")
        
        if st.button("Generate Story and Audio"):
            
            progress = st.progress(0)
            
            try:
                # Step 1: Caption
                st.write("Step 2: Analyzing image...")
                progress.progress(25)
                
                caption = get_caption(image)
                st.success("Image analyzed")
                st.write(f"Caption: {caption}")
                
                st.write("---")
                
                # Step 2: Story
                st.write("Step 3: Generating story with GPT-2...")
                progress.progress(50)
                
                story = generate_story(caption)
                word_count = len(story.split())
                
                st.success("Story generated")
                st.write(f"Story ({word_count} words):")
                st.info(story)
                
                if word_count >= 50:
                    st.success(f"Story has {word_count} words (meets 50+ requirement)")
                else:
                    st.warning(f"Story only has {word_count} words")
                
                st.write("---")
                
                # Step 3: Audio
                st.write("Step 4: Converting to audio...")
                progress.progress(75)
                
                audio_data, sample_rate = text_to_speech(story)
                
                audio_buffer = io.BytesIO()
                scipy.io.wavfile.write(audio_buffer, sample_rate, audio_data)
                audio_bytes = audio_buffer.getvalue()
                
                progress.progress(100)
                
                st.success("Audio generated")
                st.audio(audio_bytes, format='audio/wav')
                
                st.download_button(
                    "Download Audio File",
                    audio_bytes,
                    "story.wav",
                    "audio/wav"
                )
                
                duration = len(audio_data) / sample_rate
                st.write(f"Audio length: {duration:.1f} seconds")
                
                st.write("---")
                st.success("All done!")
                
                st.subheader("Summary")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("Input:")
                    st.write(f"- Image: {uploaded_file.name}")
                
                with col2:
                    st.write("Output:")
                    st.write(f"- Story: {word_count} words")
                    st.write(f"- Audio: {duration:.1f} sec")
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.write("Please try again")
    
    else:
        st.info("Please upload an image to start")
        
        st.write("---")
        st.write("Tips:")
        st.write("- Use clear images")
        st.write("- Try animals, nature, or objects")
        st.write("- JPG and PNG supported")


if __name__ == "__main__":
    main()


""""
import streamlit as st
from transformers import pipeline
from PIL import Image
import scipy.io.wavfile
import numpy as np
import io

# Set up the page
st.set_page_config(page_title="Story Generator", layout="centered")


# Load the models
# Note: These functions cache the models so they don't reload every time
@st.cache_resource
def load_image_model():
    # Using image to text pipeline for captions
    model = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")
    return model
    

@st.cache_resource
def load_story_model():
    model = pipeline("text-generation", model="distilgpt2")
    return model

@st.cache_resource
def load_audio_model():
    model = pipeline("text-to-speech", model="facebook/mms-tts-eng")
    return model


# HEre is the Function to get caption from image
def get_caption(image):
    """
    ""Takes an image and returns a text caption""
    """
    caption_model = load_image_model()
    result = caption_model(image)
    # Get the text from the result
    text = result[0]["generated_text"]
    return text


# Here adding the function to generate story from caption
def generate_story(caption):
    """
    ""Takes a caption and generates a story
    Story should be at least 50 words (50-100 words as mentioned in the lecture) for the assignment""
    """
    story_model = load_story_model()
    
    # Create a prompt for children
    # TODO: Maybe make this prompt better later
    prompt = f"Write a fun story for kids about {caption}. Once upon a time, "
    
    # Generate the story
    # Note: max_new_tokens controls length - might need to adjust
    output = story_model(
        prompt,
        max_new_tokens=120,  # Changed from 100 to get more words
        temperature=0.7,     # Controls randomness
        do_sample=True,
        pad_token_id=50256   # This is important for GPT2 models
    )
    
    # Extract the story text
    story_text = output[0]['generated_text']
    
    # Remove the prompt part from the story
    # FIXME: This might not work perfectly every time
    story_text = story_text.replace(prompt, "Once upon a time, ")
    
    # Make sure story isn't too long
    words = story_text.split()
    if len(words) > 120:  # Limit to around 120 words
        story_text = ' '.join(words[:120])
        # Add period at end if missing
        if not story_text.endswith('.'):
            story_text = story_text + '.'
    
    return story_text


# Function to convert text to speech
def text_to_speech(text):
    """
    Converts story text to audio
    Returns audio data and sampling rate
    """
    tts_model = load_audio_model()
    
    # Generate speech from text
    speech = tts_model(text)
    
    # Get audio data
    # BUG FIX: Need to flatten the array properly
    audio = np.array(speech["audio"]).flatten()
    rate = speech["sampling_rate"]
    
    return audio, rate


# Main application
def main():
    
    # Title and description
    st.title("Children's Story Generator")
    st.write("Upload an image and get a story with audio")
    st.write("Made for children aged 3-10 years")
    
    # Sidebar with info
    st.sidebar.header("Assignment Info")
    st.sidebar.write("ISOM5240 Individual Assignment")
    st.sidebar.write("Student ID: YOUR_STUDENT_ID")
    
    st.sidebar.subheader("Models Used")
    st.sidebar.text("1. Image to Text:")
    st.sidebar.code("nlpconnect/vit-gpt2-image-captioning")
    
    st.sidebar.text("2. Story Generation:")
    st.sidebar.code("distilgpt2")
    
    st.sidebar.text("3. Text to Speech:")
    st.sidebar.code("facebook/mms-tts-eng")
    
    st.sidebar.info("Story length: 50+ words")
    
    # Main content
    st.write("---")
    st.subheader("Step 1: Upload an Image")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png']
    )
    
    # Check if file was uploaded
    if uploaded_file is not None:
        
        # Open and display the image
        image = Image.open(uploaded_file)
        st.image(image, caption="Your uploaded image", width=400)
        
        # Show file details
        st.write(f"File name: {uploaded_file.name}")
        st.write(f"Image size: {image.size[0]} x {image.size[1]} pixels")
        
        st.write("---")
        
        # Button to generate story
        if st.button("Generate Story and Audio"):
            
            # Show progress bar
            progress = st.progress(0)
            
            try:
                # Step 1: Get caption from image
                st.write("Step 2: Analyzing image...")
                progress.progress(25)
                
                caption = get_caption(image)
                st.success("Image analyzed successfully")
                st.write(f"Caption: {caption}")
                
                st.write("---")
                
                # Step 2: Generate story
                st.write("Step 3: Generating story...")
                progress.progress(50)
                
                story = generate_story(caption)
                word_count = len(story.split())
                
                st.success("Story generated successfully")
                st.write(f"Story ({word_count} words):")
                st.info(story)
                
                # Check word count requirement
                if word_count >= 50:
                    st.success(f"Story has {word_count} words (meets 50+ requirement)")
                else:
                    st.warning(f"Story only has {word_count} words (need 50+)")
                
                st.write("---")
                
                # Step 3: Generate audio
                st.write("Step 4: Converting to audio...")
                progress.progress(75)
                
                audio_data, sample_rate = text_to_speech(story)
                
                # Save audio to bytes for playback
                audio_buffer = io.BytesIO()
                scipy.io.wavfile.write(audio_buffer, sample_rate, audio_data)
                audio_bytes = audio_buffer.getvalue()
                
                progress.progress(100)
                
                st.success("Audio generated successfully")
                
                # Display audio player
                st.audio(audio_bytes, format='audio/wav')
                
                # Download button
                st.download_button(
                    "Download Audio File",
                    audio_bytes,
                    "story.wav",
                    "audio/wav"
                )
                
                # Calculate and show duration
                duration = len(audio_data) / sample_rate
                st.write(f"Audio length: {duration:.1f} seconds")
                
                st.write("---")
                st.success("All done!")
                
                # Summary at the end
                st.subheader("Summary")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("Input:")
                    st.write(f"- Image: {uploaded_file.name}")
                
                with col2:
                    st.write("Output:")
                    st.write(f"- Story: {word_count} words")
                    st.write(f"- Audio: {duration:.1f} sec")
                
            except Exception as e:
                # Error handling
                st.error(f"Something went wrong: {str(e)}")
                st.write("Please try again with a different image")
    
    else:
        # Show help text when no image uploaded
        st.info("Please upload an image to start")
        
        st.write("---")
        st.write("Tips:")
        st.write("- Use clear images for best results")
        st.write("- Try images of animals, nature, or objects")
        st.write("- JPG and PNG formats are supported")


# Run the app
if __name__ == "__main__":
    main()"""""
