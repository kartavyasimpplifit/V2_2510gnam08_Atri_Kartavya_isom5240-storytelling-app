"""
ISOM5240 Assignment - AI Essay Grading Application
Student ID: 2510gnam08, S029
Name: Kartavya Atri, NUS Singapore
Target: Secondary School Chinese Essays
Description: Two-pipeline system with 3-Class Grading (Excellent, Good, Needs Improvement).
"""

import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import random
import numpy as np

# Page Config
st.set_page_config(page_title="AI Essay Grader", layout="centered")

# ==========================================
# 1. MODEL LOADING (CACHED)
# ==========================================

@st.cache_resource
def load_grading_model():
    """
    Pipeline 1: Grading Model
    Model: MirandaZhao/Finetuned_Essay_Scoring_Model_Epoch3
    """
    model_id = "MirandaZhao/Finetuned_Essay_Scoring_Model_Epoch3"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        # We explicitly tell the model to expect 3 labels to avoid shape mismatch errors
        model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=3, ignore_mismatched_sizes=True)
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading Grading Model: {e}")
        return None, None

@st.cache_resource
def load_feedback_model():
    """
    Pipeline 2: Feedback Context Model
    Model: hfl/chinese-macbert-base
    """
    model_id = "hfl/chinese-macbert-base"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSequenceClassification.from_pretrained(model_id)
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading Feedback Model: {e}")
        return None, None

# Load models
grading_tokenizer, grading_model = load_grading_model()
feedback_tokenizer, feedback_model = load_feedback_model()

# ==========================================
# 2. CORE FUNCTIONS
# ==========================================

def get_grade_and_score(text):
    """
    Pipeline 1 Logic: Determines Category and Score Range based on 3 Classes.
    IMPORTANT: We assume the model was trained with:
    0 = Needs Improvement
    1 = Good
    2 = Excellent
    """
    inputs = grading_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = grading_model(**inputs)
    
    # Get the predicted class ID (0, 1, or 2)
    logits = outputs.logits
    pred_id = torch.argmax(logits, dim=-1).item()
    
    # --- 3-CLASS MAPPING LOGIC ---
    # Adjust this if your specific training had a different order (e.g. 0=Excellent)
    if pred_id == 2: 
        category = "Excellent"
        score = random.randint(85, 100)
    elif pred_id == 1: 
        category = "Good"
        score = random.randint(60, 84)
    else: # pred_id == 0
        category = "Needs Improvement"
        score = random.randint(0, 59)
        
    return category, score

def analyze_for_feedback(text, category):
    """
    Pipeline 2 Logic: 
    Returns specific feedback and improvements based on the 3
