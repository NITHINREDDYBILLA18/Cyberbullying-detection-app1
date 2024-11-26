import requests
import gradio as gr
from flask import Flask, request, jsonify
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the saved model and tokenizer
with open('classification_model', 'rb') as f:
    classified_model = pickle.load(f)

with open('tokenized_data', 'rb') as f:
    tokenizer = pickle.load(f)

# Initialize Flask app
app = Flask(__name__)

# Define the Gradio function
def detect_cyberbullying(comment):
    try:
        # Tokenize the input text using the loaded tokenizer
        sequences = tokenizer.texts_to_sequences([comment])  # Convert text to sequences of integers
        
        # Pad the sequences to ensure uniform input size
        padded_data = pad_sequences(sequences, maxlen=150, padding='pre')  # Ensure max_len is consistent with your model's training
        
        # Predict using the model
        prediction = classified_model.predict(padded_data)

        # Return a user-friendly result based on the model's prediction
        if prediction[0] > 0.5:
            return "🚨 Warning: Cyberbullying Detected!"
        else:
            return "✅ Safe: No Cyberbullying Detected."
    except Exception as e:
        return f"Error: {str(e)}"

# Gradio Interface (with CSS for styling)
with gr.Blocks(css="""
    /* Center everything */
    .gradio-container {
        text-align: center;
        padding: 40px;
    }
    .gr-row {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    .gr-column {
        padding: 20px;
    }
    .comment-box {
        font-size: 16px;
        padding: 8px;
    }
    
    /* Title Glow Effect */
    h1 {
        font-size: 36px;
        color: #ECF0F1;
        font-weight: bold;
        text-align: center;
        text-shadow: 0 0 15px rgba(0, 204, 255, 0.8), 0 0 30px rgba(0, 204, 255, 0.6);
    }

    h3 {
        color: #BDC3C7;
        font-size: 20px;
        text-align: center;
    }
    .result-text {
        font-size: 18px;
        font-weight: bold;
        color: #ECF0F1;
    }
    .footer-text {
        font-size: 14px;
        color: #BDC3C7;
        margin-top: 20px;
    }
    /* Button Styling */
    .gr-button { 
        background-color: #2980B9; 
        color: white; 
        border-radius: 8px; 
        padding: 12px;
        font-size: 18px;
        transition: background-color 0.3s;
    }
    .gr-button:hover { 
        background-color: #3498DB; 
    }
""") as interface:
    # Header Section
    gr.Markdown(
        """
        # 🛡️ Cyberbullying Detection System  
        ### Analyze comments for harmful language and prevent online abuse.
        """
    )
    
    # Two-column Layout
    with gr.Row():
        # Input Column
        with gr.Column():
            gr.Markdown("### 💬 Enter the Comment:")
            comment_input = gr.Textbox(
                label="Comment",
                placeholder="Type a comment to analyze...",
                lines=4,
                elem_classes="comment-box"
            )
            analyze_button = gr.Button("🚀 Analyze Comment")
        
        # Output Column
        with gr.Column():
            gr.Markdown("### 🧾 Detection Result:")
            result_output = gr.Textbox(
                label="Analysis Result",
                placeholder="The result will appear here.",
                interactive=False,
                elem_classes="comment-box"
            )
    
    # Footer Section
    gr.Markdown(
        """
        ---
        **🔒 Note**: This tool is designed to assist in detecting harmful language.  
        """
    )

    # Link button, input, and output
    analyze_button.click(detect_cyberbullying, inputs=comment_input, outputs=result_output)

# Run Flask alongside Gradio
@app.route("/")
def home():
    return "Flask & Gradio deployed!"

interface.launch(share=True, server_name="0.0.0.0", server_port=7860)  # Use a different port, e.g., 7860

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)