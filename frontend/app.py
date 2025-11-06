import streamlit as st
import requests
import base64
from PIL import Image
import io

# Backend API URL
BACKEND_URL = "http://backend:8000"

# Page configuration
st.set_page_config(
    page_title="Image Segmentation with Text Prompts",
    page_icon="🎯",
    layout="wide"
)

# Title and description
st.title("🎯 Image Segmentation with Text Prompts")
st.markdown("""
This application uses **GroundingDINO** and **SAM2** to segment objects in images based on natural language descriptions.

**How it works:**
1. Upload an image
2. Describe the objects you want to segment (e.g., "cat", "person with red shirt", "all trees")
3. The system will detect and segment the objects matching your description
""")

# Sidebar for configuration
st.sidebar.header("⚙️ Configuration")

# Check backend health
try:
    health_response = requests.get(f"{BACKEND_URL}/health", timeout=5)
    if health_response.status_code == 200:
        st.sidebar.success("✅ Backend connected")
    else:
        st.sidebar.error("❌ Backend unhealthy")
except Exception as e:
    st.sidebar.error(f"❌ Backend unreachable: {str(e)}")

# Queue status
try:
    queue_response = requests.get(f"{BACKEND_URL}/queue-status", timeout=5)
    if queue_response.status_code == 200:
        queue_data = queue_response.json()
        st.sidebar.metric("GroundingDINO Queue", queue_data['grounding_dino_queue'])
        st.sidebar.metric("SAM2 Queue", queue_data['sam2_queue'])
except:
    pass

# Advanced settings
with st.sidebar.expander("Advanced Settings"):
    box_threshold = st.slider(
        "Box Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.35,
        step=0.05,
        help="Confidence threshold for object detection (lower = more detections)"
    )

    text_threshold = st.slider(
        "Text Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.25,
        step=0.05,
        help="Confidence threshold for text matching (lower = more lenient matching)"
    )

# Main content
col1, col2 = st.columns(2)

with col1:
    st.header("📤 Input")

    # File uploader
    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG, PNG"
    )

    # Text prompt
    text_prompt = st.text_input(
        "Text Prompt",
        placeholder="e.g., 'cat', 'person with blue shirt', 'all books'",
        help="Describe the objects you want to segment. Separate multiple objects with periods (e.g., 'cat . dog . bird .')"
    )

    # Display uploaded image
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

    # Process button
    if st.button("🚀 Segment Image", type="primary", disabled=(uploaded_file is None or not text_prompt)):
        with st.spinner("Processing... This may take a moment."):
            try:
                # Prepare request
                files = {
                    'image': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
                }
                data = {
                    'text_prompt': text_prompt,
                    'box_threshold': box_threshold,
                    'text_threshold': text_threshold
                }

                # Send request to backend
                response = requests.post(
                    f"{BACKEND_URL}/segment",
                    files=files,
                    data=data,
                    timeout=120
                )

                if response.status_code == 200:
                    result = response.json()
                    st.session_state['result'] = result
                    st.success("✅ Segmentation completed!")
                else:
                    st.error(f"❌ Error: {response.json().get('detail', 'Unknown error')}")

            except requests.exceptions.Timeout:
                st.error("❌ Request timed out. The image might be too large or the service is busy.")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

with col2:
    st.header("📊 Results")

    # Display results if available
    if 'result' in st.session_state:
        result = st.session_state['result']

        # Display metrics
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Objects Detected", result['detected_objects'])
        with col_b:
            st.metric("Task ID", result['task_id'][:8] + "...")

        # Display detected objects
        if result['detected_objects'] > 0:
            st.subheader("Detected Objects")

            # Create a table of detected objects
            objects_data = []
            for i, (label, score) in enumerate(zip(result['labels'], result['scores'])):
                objects_data.append({
                    'Index': i + 1,
                    'Label': label,
                    'Confidence': f"{score:.2%}"
                })

            st.table(objects_data)

            # Display segmentation visualization
            st.subheader("Segmentation Visualization")
            if result['visualization']:
                viz_bytes = base64.b64decode(result['visualization'])
                viz_image = Image.open(io.BytesIO(viz_bytes))
                st.image(viz_image, caption="Segmented Image", use_column_width=True)

                # Download button
                st.download_button(
                    label="📥 Download Result",
                    data=viz_bytes,
                    file_name=f"segmented_{uploaded_file.name if uploaded_file else 'image.png'}",
                    mime="image/png"
                )
        else:
            st.info("ℹ️ No objects detected matching the text prompt. Try adjusting the thresholds or using a different prompt.")
    else:
        st.info("👈 Upload an image and enter a text prompt to begin segmentation.")

# Examples section
with st.expander("💡 Example Prompts"):
    st.markdown("""
    Here are some example text prompts you can try:

    - **Simple objects**: `cat`, `dog`, `person`, `car`, `tree`
    - **Multiple objects**: `cat . dog . bird .`
    - **Descriptive**: `person with red shirt`, `cat on the table`, `white flowers`
    - **Positional**: `leftmost person`, `front-most car`
    - **Attributes**: `green apple`, `large building`, `old tree`

    **Tips:**
    - Separate multiple object types with periods (.)
    - Be specific for better results
    - Adjust thresholds if you get too many/few detections
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Powered by <strong>GroundingDINO</strong> and <strong>SAM2</strong></p>
    <p>Built with Streamlit, FastAPI, and Docker</p>
</div>
""", unsafe_allow_html=True)
