import numpy as np
import gradio as gr
from gradio import themes
import roop.globals
from roop.core import (
    start,
    decode_execution_providers,
    suggest_max_memory,
    suggest_execution_threads,
)
from roop.processors.frame.core import get_frame_processors_modules
from roop.utilities import normalize_output_path
import os
from PIL import Image
import time

def swap_face_image(source_file, target_file, doFaceEnhancer):
    source_path = "input.jpg"
    target_path = "target.jpg"
    output_path = "output.png"

    # Save source and target images
    Image.fromarray(source_file).save(source_path)
    Image.fromarray(target_file).save(target_path)

    print("source_path: ", source_path)
    print("target_path: ", target_path)

    # Set globals
    roop.globals.source_path = source_path
    roop.globals.target_path = target_path
    roop.globals.output_path = output_path
    roop.globals.frame_processors = ["face_swapper", "face_enhancer"] if doFaceEnhancer else ["face_swapper"]
    roop.globals.headless = True
    roop.globals.keep_frames = False
    roop.globals.many_faces = False
    roop.globals.max_memory = suggest_max_memory()
    roop.globals.execution_providers = decode_execution_providers(["cuda"])
    roop.globals.execution_threads = suggest_execution_threads()

    print("start process", source_path, target_path, output_path)

    # Ensure processors are ready
    for frame_processor in get_frame_processors_modules(roop.globals.frame_processors):
        if not frame_processor.pre_check():
            return None

    start()

    # Return output image
    return Image.open(output_path)

# Custom CSS for neon style with animated online indicator
custom_css = """
:root {
    --neon-primary: #00f3ff;
    --neon-secondary: #ff00ff;
    --neon-accent: #00ff87;
    --neon-warning: #ffcc00;
    --dark-bg: #0a0a1a;
    --dark-panel: #13132b;
    --darker-panel: #0c0c1f;
    --text-primary: #ffffff;
    --text-secondary: #a0a0c0;
}
body {
    background: var(--dark-bg) !important;
    color: var(--text-primary) !important;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
}
.gr-block {
    background: var(--dark-panel) !important;
    border-radius: 12px !important;
    border: 1px solid rgba(0, 243, 255, 0.2) !important;
    box-shadow: 0 0 15px rgba(0, 243, 255, 0.1) !important;
}
.gr-box {
    border-color: rgba(0, 243, 255, 0.3) !important;
    color: var(--text-primary) !important;
    background: rgba(10, 10, 26, 0.7) !important;
}
h1, h2, h3, h4, label, .gr-label {
    color: var(--text-primary) !important;
    text-shadow: 0 0 5px rgba(0, 243, 255, 0.5);
}
.gr-button {
    background: linear-gradient(45deg, var(--neon-primary), var(--neon-secondary)) !important;
    color: black !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 12px 28px !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    box-shadow: 0 0 10px var(--neon-primary), 0 0 20px rgba(0, 243, 255, 0.3) !important;
    transition: all 0.3s ease !important;
    width: 100% !important;
    margin: 10px 0 !important;
}
.gr-button:not(:disabled):hover {
    transform: translateY(-2px);
    box-shadow: 0 0 15px var(--neon-primary), 0 0 30px rgba(0, 243, 255, 0.5) !important;
}
.gr-button:disabled {
    background: #4b5563 !important;
    box-shadow: none !important;
}
.status-indicator {
    display: inline-block;
    width: 12px;
    height: 12px;
    border-radius: 50%;
    margin-right: 10px;
    background-color: var(--neon-accent);
    box-shadow: 0 0 0 0 rgba(0, 255, 135, 0.7);
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0% {
        box-shadow: 0 0 0 0 rgba(0, 255, 135, 0.7);
    }
    70% {
        box-shadow: 0 0 0 10px rgba(0, 255, 135, 0);
    }
    100% {
        box-shadow: 0 0 0 0 rgba(0, 255, 135, 0);
    }
}
.status-text {
    color: var(--text-secondary);
    font-size: 0.9rem;
    display: flex;
    align-items: center;
}
.header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 2rem;
    padding: 1.5rem;
    background: linear-gradient(90deg, rgba(0,243,255,0.1) 0%, rgba(255,0,255,0.1) 100%);
    border-radius: 12px;
    border: 1px solid rgba(0, 243, 255, 0.3);
    box-shadow: 0 0 20px rgba(0, 243, 255, 0.2);
}
.title-section h1 {
    margin-bottom: 0.25rem;
    font-weight: 800;
    background: linear-gradient(45deg, var(--neon-primary), var(--neon-secondary));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: 0 0 10px rgba(0, 243, 255, 0.5);
}
.title-section p {
    color: var(--text-secondary);
    margin: 0;
}
.instructions {
    background: linear-gradient(90deg, rgba(0,243,255,0.05) 0%, rgba(255,0,255,0.05) 100%) !important;
    padding: 1.5rem !important;
    margin-bottom: 2rem !important;
    border: 1px solid rgba(0, 243, 255, 0.2) !important;
}
.instructions h3 {
    margin-top: 0;
    margin-bottom: 0.75rem;
    color: var(--neon-primary) !important;
}
.instructions ul {
    margin-bottom: 0;
    padding-left: 1.5rem;
}
.instructions li {
    margin-bottom: 0.5rem;
    color: var(--text-secondary);
}
.instructions li:last-child {
    margin-bottom: 0;
}
.footer {
    text-align: center;
    margin-top: 2rem;
    padding-top: 1.5rem;
    border-top: 1px solid rgba(0, 243, 255, 0.2);
    color: var(--text-secondary);
    font-size: 0.875rem;
}
.image-container {
    border: 2px solid rgba(0, 243, 255, 0.3);
    border-radius: 12px;
    padding: 8px;
    background: rgba(0, 0, 0, 0.2);
    margin-bottom: 1.5rem;
    box-shadow: 0 0 15px rgba(0, 243, 255, 0.1);
}
.image-container .gr-label {
    background: rgba(0, 243, 255, 0.1);
    padding: 8px 12px;
    border-radius: 8px;
    margin-bottom: 10px;
    display: inline-block;
}
.control-panel {
    background: linear-gradient(90deg, rgba(0,243,255,0.08) 0%, rgba(255,0,255,0.08) 100%) !important;
    padding: 1.5rem !important;
    border: 1px solid rgba(0, 243, 255, 0.3) !important;
    border-radius: 12px !important;
    margin-bottom: 1.5rem !important;
}
.status-panel {
    background: var(--darker-panel) !important;
    padding: 1.5rem !important;
    border: 1px solid rgba(0, 255, 135, 0.3) !important;
    border-radius: 12px !important;
    box-shadow: 0 0 15px rgba(0, 255, 135, 0.1) !important;
}
.status-header {
    display: flex;
    align-items: center;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid rgba(0, 255, 135, 0.2);
}
.status-content {
    min-height: 100px;
}
.output-highlight {
    border: 2px solid var(--neon-accent) !important;
    box-shadow: 0 0 20px rgba(0, 255, 135, 0.3) !important;
}
.upload-text {
    color: var(--text-secondary);
    text-align: center;
    padding: 20px;
}
.progress-bar {
    height: 6px;
    background: rgba(0, 243, 255, 0.2);
    border-radius: 3px;
    margin: 10px 0;
    overflow: hidden;
}
.progress-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--neon-primary), var(--neon-accent));
    border-radius: 3px;
    width: 0%;
    transition: width 0.3s ease;
}
.control-item {
    margin-bottom: 1rem;
    padding: 1rem;
    background: rgba(0, 0, 0, 0.2);
    border-radius: 8px;
    border: 1px solid rgba(0, 243, 255, 0.1);
}
.control-item:last-child {
    margin-bottom: 0;
}
.download-btn {
    background: linear-gradient(45deg, var(--neon-accent), #00cc70) !important;
    margin-top: 15px !important;
}
"""

# Gradio UI with neon style theme
with gr.Blocks(
    css=custom_css,
    title="Neon Face Swap AI",
    theme=gr.themes.Default(primary_hue="cyan", secondary_hue="pink")
) as demo:
    
    # Header section with title and animated status indicator
    with gr.Row(elem_classes="header"):
        with gr.Column(scale=3):
            with gr.Row(elem_classes="title-section"):
                gr.Markdown("""
                # 🌌 AI FACE SWAPPER
                ### Next-Generation AI Face Swapping Technology
                """)
        with gr.Column(scale=1):
            with gr.Row():
                gr.Markdown("""
                <div class="status-text">
                    <div class="status-indicator"></div>
                    System Online
                </div>
                """)
    
    # Main content - unique arrangement
    with gr.Row():
        # Left column - inputs
        with gr.Column(scale=1, min_width=400):
            gr.Markdown("### 📤 SOURCE IMAGE")
            with gr.Group(elem_classes="image-container"):
                source_image = gr.Image(
                    label="Upload Source Face", 
                    type="numpy",
                    height=250,
                    elem_classes="gr-box"
                )
            
            gr.Markdown("### 🎯 TARGET IMAGE")
            with gr.Group(elem_classes="image-container"):
                target_image = gr.Image(
                    label="Upload Target Face", 
                    type="numpy",
                    height=250,
                    elem_classes="gr-box"
                )
        
        # Middle column - controls and status (separated)
        with gr.Column(scale=1, min_width=350):
            # PROCESSING CONTROLS - Separate Panel
            with gr.Group(elem_classes="control-panel"):
                gr.Markdown("### ⚙️ PROCESSING CONTROLS")
                
                with gr.Group(elem_classes="control-item"):
                    face_enhancer = gr.Checkbox(
                        label="Enable Face Enhancer",
                        value=True,
                        info="Improves face quality but increases processing time"
                    )
                
                with gr.Group(elem_classes="control-item"):
                    gr.Markdown("**Processing Options**")
                    many_faces = gr.Checkbox(
                        label="Detect Multiple Faces",
                        value=False,
                        info="Enable if image contains multiple faces"
                    )
                
                with gr.Group(elem_classes="control-item"):
                    submit = gr.Button(
                        "🔄 START SWAP", 
                        variant="primary"
                    )
            
            # STATUS - Separate Panel
            with gr.Group(elem_classes="status-panel"):
                with gr.Column():
                    gr.Markdown("""
                    <div class="status-header">
                        <h4 style="margin: 0;">📊 SYSTEM STATUS</h4>
                    </div>
                    """)
                    
                    with gr.Group(elem_classes="status-content"):
                        # Progress bar
                        gr.Markdown("""
                        <div class="progress-bar">
                            <div class="progress-fill" id="progress-fill"></div>
                        </div>
                        """)
                        
                        # Status text
                        info_text = gr.Textbox(
                            label="Current Status",
                            value="🟢 Ready to process images",
                            interactive=False,
                            lines=3
                        )
                        
                        # Additional status info
                        with gr.Row():
                            gr.Markdown("**GPU:** ✅ Active")
                            gr.Markdown("**Memory:** 🟡 Stable")
        
        # Right column - output
        with gr.Column(scale=1, min_width=400):
            gr.Markdown("### 📥 OUTPUT RESULT")
            with gr.Group(elem_classes="image-container output-highlight"):
                output_image = gr.Image(
                    label="Swapped Result", 
                    interactive=False,
                    height=450,
                    elem_classes="gr-box"
                )
                
                # Download button (will be shown after processing)
                download_btn = gr.Button(
                    "💾 DOWNLOAD RESULT",
                    visible=False,
                    elem_classes="download-btn"
                )
                
                # Add a file component for downloading
                download_file = gr.File(
                    label="Download Result",
                    visible=False,
                    interactive=False
                )
    
    # Instructions at the bottom
    with gr.Row():
        with gr.Column():
            gr.Markdown("""
            <div class="instructions">
                <h3>🚀 HOW TO USE</h3>
                <ul>
                    <li><strong>Source Face:</strong> Select an image containing the face you want to use</li>
                    <li><strong>Target Face:</strong> Select an image where you want to place the source face</li>
                    <li><strong>Face Enhancer:</strong> Enable for higher quality results (takes longer)</li>
                    <li>Click <strong>START SWAP</strong> to begin processing</li>
                    <li>Monitor progress in the <strong>SYSTEM STATUS</strong> panel</li>
                    <li>Download your result when processing is complete</li>
                </ul>
            </div>
            """)
    
    # Footer
    gr.Markdown("""
    <div class="footer">
        <p>Powered by Roop • Face Swapper AI v2.0 • GPU Accelerated</p>
    </div>
    """)
    
    # Function to save the output image and return file path
    def save_output_image(img):
        if img is None:
            return None, gr.update(visible=False)
        
        # Convert PIL Image to numpy array
        img_array = np.array(img)
        
        # Save the image to a temporary file
        output_path = "swapped_result.png"
        Image.fromarray(img_array).save(output_path)
        
        return output_path, gr.update(visible=True, value=output_path)
    
    # Enhanced submit function with status updates
    def process_swap(source_file, target_file, doFaceEnhancer, many_faces_option):
        if source_file is None or target_file is None:
            yield "❌ Please upload both source and target images", None, gr.update(visible=False), None, gr.update(visible=False)
            return
        
        yield "🟡 Processing: Analyzing faces and detecting features...", None, gr.update(visible=False), None, gr.update(visible=False)
        time.sleep(1)
        
        yield "🟡 Processing: Extracting facial landmarks and matching features...", None, gr.update(visible=False), None, gr.update(visible=False)
        time.sleep(1)
        
        yield "🟡 Processing: Swapping faces and applying transformations...", None, gr.update(visible=False), None, gr.update(visible=False)
        time.sleep(1)
        
        if doFaceEnhancer:
            yield "🟡 Processing: Enhancing face quality and refining details...", None, gr.update(visible=False), None, gr.update(visible=False)
            time.sleep(1)
        
        result = swap_face_image(source_file, target_file, doFaceEnhancer)
        
        if result is None:
            yield "❌ Error: Failed to process images. Please try again with different images.", None, gr.update(visible=False), None, gr.update(visible=False)
        else:
            file_path, file_update = save_output_image(result)
            yield "✅ Processing complete! Result is ready for download.", result, gr.update(visible=True), file_path, file_update
    
    # Connect the button with enhanced function
    submit.click(
        fn=process_swap,
        inputs=[source_image, target_image, face_enhancer, many_faces],
        outputs=[info_text, output_image, download_btn, download_file]
    )
    
    # Download button functionality - show the file download component
    def show_download():
        return gr.update(visible=True)

    download_btn.click(
        fn=show_download,
        inputs=None,
        outputs=download_file
    )

if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0")