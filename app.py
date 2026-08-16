import streamlit as st
import nibabel as nib
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import tempfile
from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt
import time
import gdown
from pathlib import Path
import tensorflow as tf
from scipy.ndimage import zoom

# Force CPU-only operation to avoid CUDA errors
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Set page config
st.set_page_config(page_title="Glioma Segmentation", layout="wide")

# Initialize scaler
scaler = MinMaxScaler()
# Constants
MODEL_URL = "https://drive.google.com/uc?id=15DvYjyBHo-OgI-oVPrruocNr_WUauQEk"
MODEL_DIR = "saved_model"
MODEL_PATH = os.path.join(MODEL_DIR, "3D_unet_100_epochs_2_batch_patch_training.keras")

# Model expects (96, 96, 96, 4) input - UPDATED based on common 3D U-Net architectures
TARGET_SHAPE = (96, 96, 96, 4)

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Download model from Google Drive (cache this to avoid repeated downloads)
@st.cache_resource
def download_and_load_model():
    # Check if model already exists
    if not os.path.exists(MODEL_PATH):
        st.info("Downloading model from Google Drive... (This may take a few minutes)")
        try:
            # Download using gdown
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
            
            # Verify download
            if not os.path.exists(MODEL_PATH):
                raise FileNotFoundError("Model download failed")
                
        except Exception as e:
            st.error(f"Failed to download model: {str(e)}")
            return None
    
    # Load the model
    try:
        # Disable TensorFlow logging
        tf.get_logger().setLevel('ERROR')
        
        # Load model with custom objects if needed
        model = load_model(MODEL_PATH, compile=False)
        
        # Debug: Print model input shape
        st.write(f"Model input shape: {model.input_shape}")
        st.write(f"Model output shape: {model.output_shape}")
        
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Function to process uploaded files
def process_uploaded_files(uploaded_files):
    modalities = {}
    
    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name.lower()
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.nii.gz') as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            tmp_path = tmp_file.name
        
        try:
            # Load NIfTI file
            img = nib.load(tmp_path)
            img_data = img.get_fdata()
            
            # Scale the data
            img_data = scaler.fit_transform(img_data.reshape(-1, img_data.shape[-1])).reshape(img_data.shape)
            
            # Determine modality
            if 't1n' in file_name or 't1' in file_name and 't1c' not in file_name:
                modalities['t1n'] = img_data
            elif 't1c' in file_name or 't1ce' in file_name:
                modalities['t1c'] = img_data
            elif 't2f' in file_name or 'flair' in file_name:
                modalities['t2f'] = img_data
            elif 't2w' in file_name or 't2' in file_name:
                modalities['t2w'] = img_data
            elif 'seg' in file_name or 'mask' in file_name:
                modalities['mask'] = img_data.astype(np.uint8)
                
        except Exception as e:
            st.error(f"Error processing file {uploaded_file.name}: {str(e)}")
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    return modalities

# Function to prepare input for model
def prepare_input(modalities):
    # Check we have all required modalities
    required = ['t1n', 't1c', 't2f', 't2w']
    if not all(m in modalities for m in required):
        missing = [m for m in required if m not in modalities]
        st.error(f"Missing modalities: {missing}")
        return None, None, None
    
    # Combine modalities
    combined = np.stack([
        modalities['t1n'],
        modalities['t1c'],
        modalities['t2f'],
        modalities['t2w']
    ], axis=3)
    
    # Get original shape
    original_shape = combined.shape
    
    # Center crop to (96, 96, 96, 4) - common size for 3D U-Net
    depth, height, width, channels = combined.shape
    
    # Calculate crop coordinates
    start_depth = max((depth - TARGET_SHAPE[0]) // 2, 0)
    start_height = max((height - TARGET_SHAPE[1]) // 2, 0)
    start_width = max((width - TARGET_SHAPE[2]) // 2, 0)
    
    end_depth = min(start_depth + TARGET_SHAPE[0], depth)
    end_height = min(start_height + TARGET_SHAPE[1], height)
    end_width = min(start_width + TARGET_SHAPE[2], width)
    
    # Crop the volume
    cropped = combined[start_depth:end_depth, 
                      start_height:end_height, 
                      start_width:end_width, :]
    
    # If cropped volume is smaller than target, pad it
    if cropped.shape[:3] != TARGET_SHAPE[:3]:
        pad_depth = TARGET_SHAPE[0] - cropped.shape[0]
        pad_height = TARGET_SHAPE[1] - cropped.shape[1]
        pad_width = TARGET_SHAPE[2] - cropped.shape[2]
        
        cropped = np.pad(cropped, 
                        ((0, pad_depth), (0, pad_height), (0, pad_width), (0, 0)),
                        mode='constant')
    
    return cropped, original_shape, combined

# Function to make prediction
def make_prediction(model, input_data):
    # Add batch dimension
    input_data = np.expand_dims(input_data, axis=0)
    
    # Make prediction
    prediction = model.predict(input_data, verbose=0)
    
    # Handle different output formats
    if len(prediction.shape) == 5:  # (batch, x, y, z, classes)
        prediction_argmax = np.argmax(prediction, axis=4)[0, :, :, :]
    else:  # (batch, x, y, z) - already argmaxed
        prediction_argmax = prediction[0, :, :, :]
    
    return prediction_argmax

# Function to upsample prediction to original size
def upsample_prediction(prediction, target_shape):
    # Get the crop coordinates used in prepare_input
    depth, height, width = target_shape[:3]
    
    # Create empty array for full size prediction
    full_prediction = np.zeros(target_shape[:3], dtype=prediction.dtype)
    
    # Calculate where to place the prediction
    start_depth = max((depth - prediction.shape[0]) // 2, 0)
    start_height = max((height - prediction.shape[1]) // 2, 0)
    start_width = max((width - prediction.shape[2]) // 2, 0)
    
    end_depth = start_depth + prediction.shape[0]
    end_height = start_height + prediction.shape[1]
    end_width = start_width + prediction.shape[2]
    
    # Ensure we don't exceed bounds
    end_depth = min(end_depth, depth)
    end_height = min(end_height, height)
    end_width = min(end_width, width)
    
    # Place the prediction in the correct location
    full_prediction[start_depth:end_depth, 
                   start_height:end_height, 
                   start_width:end_width] = prediction[:end_depth-start_depth, 
                                                     :end_height-start_height, 
                                                     :end_width-start_width]
    
    return full_prediction

# Function to visualize results
def visualize_results(original_data, prediction, ground_truth=None):
    # Select a modality to display (using T1c here)
    image_data = original_data[:, :, :, 1]  # T1c is the second channel
    
    # Select some slices to display (middle slices)
    depth = image_data.shape[2]
    slice_indices = [depth//4, depth//2, 3*depth//4]
    
    # Create figure
    fig, axes = plt.subplots(3, 3 if ground_truth is not None else 2, 
                            figsize=(12, 8))
    
    for i, slice_idx in enumerate(slice_indices):
        # Rotate images for better visualization
        img_slice = np.rot90(image_data[:, :, slice_idx])
        pred_slice = np.rot90(prediction[:, :, slice_idx])
        
        # Plot input image
        axes[i, 0].imshow(img_slice, cmap='gray')
        axes[i, 0].set_title(f'Input Image - Slice {slice_idx}')
        axes[i, 0].axis('off')
        
        # Plot prediction
        axes[i, 1].imshow(pred_slice, cmap='viridis')
        axes[i, 1].set_title(f'Prediction - Slice {slice_idx}')
        axes[i, 1].axis('off')
        
        # Plot ground truth if available
        if ground_truth is not None:
            gt_slice = np.rot90(ground_truth[:, :, slice_idx])
            axes[i, 2].imshow(gt_slice, cmap='viridis')
            axes[i, 2].set_title(f'Ground Truth - Slice {slice_idx}')
            axes[i, 2].axis('off')
    
    plt.tight_layout()
    return fig

# Main app
def main():
    st.title("3D Glioma Segmentation with U-Net")
    st.write("Upload MRI scans in NIfTI format for glioma segmentation")
    
    with st.expander("How to use this app"):
        st.markdown("""
        1. Upload **all four MRI modalities** (T1n, T1c, T2f, T2w) as NIfTI files (.nii.gz)
        2. Optionally upload a segmentation mask for comparison (must contain 'seg' in filename)
        3. Click 'Process and Predict' button
        4. View the segmentation results
        
        **Note:** 
        - The first run will download the model (~100MB) which may take a few minutes.
        - This version runs on CPU and may be slower than GPU-accelerated versions.
        - Supported file naming: t1n/t1, t1c/t1ce, t2f/flair, t2w/t2
        """)
    
    # Load model (this will trigger download if needed)
    model = download_and_load_model()
    
    if model is None:
        st.error("Failed to load model. Please check the error message above.")
        return
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Upload MRI scans (NIfTI format)",
        type=['nii', 'nii.gz'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.write(f"Uploaded {len(uploaded_files)} files")
        
        # Show uploaded files for debugging
        for file in uploaded_files:
            st.write(f"- {file.name}")
    
    if uploaded_files and len(uploaded_files) >= 4:
        if st.button("Process and Predict"):
            with st.spinner("Processing files..."):
                # Process uploaded files
                modalities = process_uploaded_files(uploaded_files)
                
                # Debug: show which modalities were found
                st.write(f"Found modalities: {list(modalities.keys())}")
                
                # Prepare input
                input_data, original_shape, original_data = prepare_input(modalities)
                
                if input_data is None:
                    st.error("Could not prepare input data. Please ensure you've uploaded all required modalities.")
                    return
                
                st.write(f"Input data shape: {input_data.shape}")
                st.write(f"Expected input shape: {TARGET_SHAPE}")
                
                # Get ground truth if available
                ground_truth = None
                if 'mask' in modalities:
                    ground_truth = modalities['mask']
                    if ground_truth.max() == 4:  # Handle different label conventions
                        ground_truth[ground_truth == 4] = 3
                
                # Make prediction
                with st.spinner("Making prediction (this may take a few minutes on CPU)..."):
                    start_time = time.time()
                    try:
                        prediction = make_prediction(model, input_data)
                        
                        # Upsample prediction to original size
                        prediction = upsample_prediction(prediction, original_shape)
                        
                        # Convert prediction to int32 for NIfTI compatibility
                        prediction = prediction.astype(np.int32)
                        
                        elapsed_time = time.time() - start_time
                        st.success(f"Prediction completed in {elapsed_time:.2f} seconds")
                        
                    except Exception as e:
                        st.error(f"Prediction failed: {str(e)}")
                        return
                
                # Visualize results
                fig = visualize_results(original_data, prediction, ground_truth)
                st.pyplot(fig)
                
                # Provide download option for prediction
                st.subheader("Download Prediction")
                
                # Create a temporary NIfTI file for download
                with tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False) as tmp_file:
                    # Convert prediction to NIfTI with explicit dtype
                    pred_img = nib.Nifti1Image(prediction, affine=np.eye(4), dtype=np.int32)
                    nib.save(pred_img, tmp_file.name)
                    
                    # Read back the file data
                    with open(tmp_file.name, 'rb') as f:
                        pred_data = f.read()
                    
                    # Clean up
                    os.unlink(tmp_file.name)
                
                st.download_button(
                    label="Download Segmentation (NIfTI)",
                    data=pred_data,
                    file_name="glioma_segmentation.nii.gz",
                    mime="application/octet-stream"
                )
    elif uploaded_files and len(uploaded_files) < 4:
        st.warning("Please upload all four modalities (T1n, T1c, T2f, T2w)")

if __name__ == "__main__":
    main()
