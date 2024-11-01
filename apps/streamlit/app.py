# app.py
import os
import time

def setup_cuda_env(sleep_second=2):
    """Initialize CUDA environment with optimal settings"""
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    time.sleep(sleep_second)  # Allow settings to take effect

# Call this before any other imports
setup_cuda_env()

import streamlit as st
import sqlite3
import tempfile
from PIL import Image
import json
from pathlib import Path
import time
from datetime import datetime
from st_aggrid import AgGrid, GridOptionsBuilder
from st_aggrid.grid_options_builder import GridOptionsBuilder
import pandas as pd
import shutil
from promptfix import (
    initialize_model, process_image
)


st.set_page_config(
    page_title='PromptFix',
    layout="wide",
    initial_sidebar_state="expanded",
)


# Database setup
DB_PATH = "promptfix.db"
IMAGES_DIR = "stored_images"

def init_database():
    """Initialize SQLite database and create tables if they don't exist"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS fixed_images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            original_img_path TEXT NOT NULL,
            fixed_img_path TEXT NOT NULL,
            task_type TEXT NOT NULL,
            fix_prompt TEXT NOT NULL,
            context_prompt TEXT,
            processing_time FLOAT,
            fix_quality TEXT CHECK(fix_quality IN ('High', 'Medium', 'Low', 'Pending')),
            note TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def save_uploaded_file(uploaded_file, directory):
    """Save uploaded file and return the saved path"""
    os.makedirs(directory, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(directory, f"{timestamp}_{uploaded_file.name}")
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def insert_fix_record(original_path, fixed_path, task_type, fix_prompt, context_prompt, processing_time):
    """Insert a new record into the fixed_images table"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        INSERT INTO fixed_images 
        (original_img_path, fixed_img_path, task_type, fix_prompt, context_prompt, 
         processing_time, fix_quality, note)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (original_path, fixed_path, task_type, fix_prompt, context_prompt, 
          processing_time, 'Pending', ''))
    fix_id = c.lastrowid
    conn.commit()
    conn.close()
    return fix_id

def update_fix_record(fix_id, quality, note):
    """Update the quality and note for a fix record"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        UPDATE fixed_images 
        SET fix_quality = ?, note = ?
        WHERE id = ?
    ''', (quality, note, fix_id))
    conn.commit()
    conn.close()

def get_all_fixes():
    """Retrieve all fix records from the database"""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query('''
        SELECT * FROM fixed_images 
        ORDER BY created_at DESC
    ''', conn)
    conn.close()
    return df

def get_task_prompts(task_type):
    """Return default prompts for each task type"""
    prompts = {
        "Colorization": (
            "Colorize this black and white image naturally.",
            "The image is in black and white"
        ),
        "Object Removal": (
            "Remove the unwanted object while maintaining background consistency.",
            "The image contains an unwanted object"
        ),
        "Dehazing": (
            "Remove the haze to improve image clarity.",
            "The image appears hazy or foggy"
        ),
        "Deblurring": (
            "Sharpen and restore this blurry image.",
            "The image is blurry or out of focus"
        ),
        "Watermark Removal": (
            "Remove the watermark while preserving the underlying image.",
            "The image contains a visible watermark"
        ),
        "Snow Removal": (
            "Remove snow from the scene while maintaining natural appearance.",
            "The image contains snow that needs to be removed"
        ),
        "Low-light Enhancement": (
            "Enhance the lighting while preserving natural colors and details.",
            "The image is too dark or underexposed"
        )
    }
    return prompts.get(task_type, ("", ""))

def prompt_fix_page():
    st.title("PromptFix Image Editor")
    
    # Initialize session state
    if 'processed_image' not in st.session_state:
        st.session_state.processed_image = None
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'current_fix_id' not in st.session_state:
        st.session_state.current_fix_id = None

    # Sidebar settings
    with st.sidebar:
        st.header("Settings")
        resolution = st.slider("Resolution", 256, 1024, 256, 64)
        steps = st.slider("Steps", 5, 50, 5, 1)
        cfg_text = st.slider("Text CFG Scale", 1.0, 10.0, 6.5, 0.1)
        cfg_image = st.slider("Image CFG Scale", 0.1, 5.0, 1.25, 0.05)
        
        with st.expander("Advanced Settings"):
            seed = st.number_input("Seed", value=2024)
            disable_hf_guidance = st.checkbox("Disable HF Guidance", value=True)
            enable_flaw_prompt = st.checkbox("Enable Flaw Prompt", value=True)

    # Main content
    col1, col2 = st.columns(2)
    
    with col1:
        task_type = st.selectbox(
            "Select Task",
            ["Colorization", "Object Removal", "Dehazing", "Deblurring",
             "Watermark Removal", "Snow Removal", "Low-light Enhancement"]
        )
        
        task_prompt = st.text_area("Task Prompt", 
            value=get_task_prompts(task_type)[0])
        context_prompt = st.text_area("Context Prompt", 
            value=get_task_prompts(task_type)[1])
        
        uploaded_file = st.file_uploader("Choose an image...", 
            type=["png", "jpg", "jpeg"])

    # Process and display results
    if uploaded_file is not None:
        # Display original image
        original_image = Image.open(uploaded_file)
        st.subheader("Original Image")
        st.image(original_image)

        if st.button("Fix Image"):
            try:
                start_time = time.time()
                
                # Save original image
                original_path = save_uploaded_file(uploaded_file, IMAGES_DIR)
                
                # Process image
                with st.spinner('Processing image...'):
                    if st.session_state.model is None:
                        st.session_state.model = initialize_model(  # Changed to initialize_model and store result
                            config_path="configs/promptfix.yaml",
                            ckpt_path="./checkpoints/promptfix.ckpt"
                        )

                    # Generate output path
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_path = os.path.join(IMAGES_DIR, 
                        f"fixed_{timestamp}_{uploaded_file.name}")
                    
                    # Process the image
                    process_image(
                        model=st.session_state.model,
                        image_path=original_path,
                        output_path=output_path,
                        task_prompt=task_prompt,
                        context_prompt=context_prompt,
                        resolution=resolution,
                        steps=steps,
                        cfg_text=cfg_text,
                        cfg_image=cfg_image,
                        seed=seed,
                        disable_hf_guidance=disable_hf_guidance,
                        enable_flaw_prompt=enable_flaw_prompt
                    )
                    
                    processing_time = time.time() - start_time
                    
                    # Save record to database
                    fix_id = insert_fix_record(
                        original_path, output_path, task_type,
                        task_prompt, context_prompt, processing_time
                    )
                    st.session_state.current_fix_id = fix_id
                    
                    # Display results
                    if os.path.exists(output_path):
                        processed_image = Image.open(output_path)
                        st.session_state.processed_image = processed_image
                        
                        st.subheader("Results")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(original_image, caption="Original")
                        with col2:
                            st.image(processed_image, caption="Processed")
                            
                        # Feedback form
                        st.subheader("Provide Feedback")
                        quality = st.selectbox("Fix Quality", 
                            ["Pending", "High", "Medium", "Low"])
                        note = st.text_area("Comments")
                        
                        if st.button("Submit Feedback"):
                            update_fix_record(fix_id, quality, note)
                            st.success("Feedback submitted successfully!")
                    
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                if "out of memory" in str(e).lower():
                    st.warning("Try reducing the resolution or number of steps.")

def fix_review_page():
    st.title("Fix Review")
    
    # Load data
    df = get_all_fixes()
    
    # Configure grid options
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_pagination(paginationAutoPageSize=True)
    gb.configure_selection('single', use_checkbox=True)
    gb.configure_column("created_at", type=["dateColumnFilter","customDateTimeFormat"], custom_format_string='yyyy-MM-dd HH:mm:ss')
    gridOptions = gb.build()
    
    # Display grid
    grid_response = AgGrid(
        df,
        gridOptions=gridOptions,
        data_return_mode='AS_INPUT',
        update_mode='MODEL_CHANGED',
        fit_columns_on_grid_load=True,
        enable_enterprise_modules=True,
        height=400,
        width='100%'
    )
    
    # Review selected record
    selected = grid_response['selected_rows']
    if selected:
        st.subheader("Review Fix")
        record = selected[0]
        
        # Display images
        col1, col2 = st.columns(2)
        with col1:
            st.image(record['original_img_path'], caption="Original Image")
        with col2:
            st.image(record['fixed_img_path'], caption="Fixed Image")
        
        # Display and edit metadata
        st.write(f"Task Type: {record['task_type']}")
        st.write(f"Processing Time: {record['processing_time']:.2f} seconds")
        st.write("Fix Prompt:", record['fix_prompt'])
        st.write("Context Prompt:", record['context_prompt'])
        
        # Update quality and notes
        new_quality = st.selectbox("Update Quality", 
            ["High", "Medium", "Low", "Pending"], 
            index=["High", "Medium", "Low", "Pending"].index(record['fix_quality']))
        new_note = st.text_area("Update Comments", value=record['note'] or "")
        
        if st.button("Update Review"):
            update_fix_record(record['id'], new_quality, new_note)
            st.success("Review updated successfully!")
            st.experimental_rerun()

def main():
    # Initialize database
    init_database()
    os.makedirs(IMAGES_DIR, exist_ok=True)
    
    # Page navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Prompt Fix", "Fix Review"])
    
    if page == "Prompt Fix":
        prompt_fix_page()
    else:
        fix_review_page()

if __name__ == "__main__":
    main()