"""
ISOM5240 Assignment - Children's Storytelling Application
Student ID: 2510gnam08, S029
Name: Kartavya Atri
Defining Target: Children aged is ~3-10 years 
"""

import streamlit as st
from transformers import pipeline
from PIL import Image
import scipy.io.wavfile
import numpy as np
import io

# ============================================================
# PAGE SETUP
# ============================================================
st.set_page_config(
    page_title="Story Generator",
    page_icon=":)",
    layout="centered"
)

# ============================================================
# Added additional CACHE MODELS, if i can save the cache (Trying) (Loads once, then cached)
# Doing all the 3 togethger, hope its okk!
# ============================================================

@st.cache_resource
def load_image_caption_model():
    """Load image-to-text model"""
    return pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")

@st.cache_resource
def load_story_model():
    """Load story generation model"""
    return pipeline("text-generation", model="distilgpt2")

@st.cache_resource
def load_audio_model():
    """Load text-to-speech model"""
    return pipeline("text-to-speech", model="espnet/kan-bayashi_ljspeech_vits")


# ============================================================
# FUNCTION 1: IMAGE TO TEXT
# ============================================================

def img2text(image):
    """
    Convert image to descriptive caption
    Model: nlpconnect/vit-gpt2-image-captioning
    """
    model = load_image_caption_model()
    result = model(image)
    text = result[0]["generated_text"]
    return text


# ============================================================
# FUNCTION 2: TEXT TO STORY
# ============================================================

def text2story(text):
    """
    Generate children's story from caption
    Model: distilgpt2
    Length: 50-100 words
    """
    model = load_story_model()
    
    prompt = f"Write a short magical story for children about {text}. Once upon a time,"
    
    result = model(
        prompt,
        max_new_tokens=120,
        temperature=0.75,
        do_sample=True,
        repetition_penalty=1.2,
        pad_token_id=50256
    )
    
    story = result[0]['generated_text']
    story = story.replace(prompt, "Once upon a time,").strip()
    
    words = story.split()
    if len(words) > 100:
        story = ' '.join(words[:100]) + '.'
    
    return story


# ============================================================
# FUNCTION 3: TEXT TO AUDIO
# ============================================================

def text2audio(story_text):
    """
    Convert story to speech
    Model: espnet/kan-bayashi_ljspeech_vits
    """
    model = load_audio_model()
    speech = model(story_text)
    
    audio_data = np.array(speech["audio"]).flatten()
    sampling_rate = speech["sampling_rate"]
    
    return audio_data, sampling_rate


# ============================================================
# MAIN APPLICATION
# ============================================================

def main():
    # Header
    st.title(" Children's Story Generator")
    st.markdown("### Transform Images into Magical Audio Stories")
    st.write("*For children aged 3-10 years*")
    
    # Sidebar
    with st.sidebar:
        st.header("About")
        st.markdown("""
        **Three-Stage AI Pipeline:**
        
        **Stage 1:** Image → Caption  
        `vit-gpt2-image-captioning`
        
        **Stage 2:** Caption → Story  
        `distilgpt2` (50-100 words)
        
        **Stage 3:** Story → Audio  
        `VITS TTS`
        """)
        
        st.info("✨ **Assignment:** ISOM5240")
        st.warning("⏱️ First run takes 3-5 minutes (loading models)")
    
    st.markdown("---")
    
    # Image upload
    st.subheader("📸 Step 1: Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an image (JPG, PNG)",
        type=['jpg', 'jpeg', 'png']
    )
    
    if uploaded_file:
        # Show image
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.image(image, caption="Your Image", use_container_width=True)
        with col2:
            st.write("**File Info:**")
            st.write(f"📁 {uploaded_file.name}")
            st.write(f"📏 {image.size[0]}×{image.size[1]} px")
        
        st.markdown("---")
        
        # Generate button
        if st.button("✨ Generate Story & Audio", type="primary"):
            
            progress = st.progress(0)
            status = st.empty()
            
            try:
                # ===== STAGE 1 =====
                st.markdown("### Stage 1: Image Analysis")
                status.text("Analyzing image...")
                progress.progress(20)
                
                caption = img2text(image)
                
                st.success(f"**Caption:** {caption}")
                progress.progress(35)
                st.write("✅ Stage 1 Complete!")
                
                st.markdown("---")
                
                # ===== STAGE 2 =====
                st.markdown("### 📖 Stage 2: Story Creation")
                status.text("Writing story...")
                progress.progress(50)
                
                story = text2story(caption)
                word_count = len(story.split())
                
                st.write("**Generated Story:**")
                st.info(story)
                st.write(f"*Length: {word_count} words*")
                
                if 50 <= word_count <= 100:
                    st.success(f"✅ Stage 2 Complete! Perfect length: {word_count} words")
                else:
                    st.warning(f"⚠️ Word count: {word_count}")
                
                progress.progress(70)
                st.markdown("---")
                
                # ===== STAGE 3 =====
                st.markdown("### 🎙️ Stage 3: Audio Generation")
                status.text("Converting to speech...")
                progress.progress(85)
                
                audio_data, sampling_rate = text2audio(story)
                
                # Save to buffer
                audio_buffer = io.BytesIO()
                scipy.io.wavfile.write(audio_buffer, sampling_rate, audio_data)
                audio_bytes = audio_buffer.getvalue()
                
                progress.progress(100)
                status.empty()
                
                # Play audio
                st.audio(audio_bytes, format='audio/wav')
                
                # Download
                st.download_button(
                    "💾 Download Audio",
                    audio_bytes,
                    "story.wav",
                    "audio/wav"
                )
                
                duration = len(audio_data) / sampling_rate
                st.success(f"✅ Stage 3 Complete! Audio: {duration:.1f}s")
                
                # Summary
                st.markdown("---")
                st.balloons()
                st.success("🎉 All stages completed successfully!")
                
                with st.expander(" View Summary"):
                    st.write(f"""
                    - **Image:** {uploaded_file.name}
                    - **Caption:** {caption}
                    - **Story Length:** {word_count} words
                    - **Audio Duration:** {duration:.1f} seconds
                    """)
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.info("Please try again or use a different image")
    
    else:
        st.info(" Upload an image to begin!")
        
        st.markdown("---")
        st.write("** Try these image types:**")
        
        col1, col2, col3 = st.columns(3)
        col1.write(" **Animals**\nPets, wildlife")
        col2.write(" **Nature**\nLandscapes, flowers")
        col3.write(" **Objects**\nToys, vehicles")


if __name__ == "__main__":
    main()
