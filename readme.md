Okay, here's a comprehensive README.md file for your project, incorporating the features, setup instructions, and usage examples:

# AI Model API Service

This project implements a FastAPI-based API for serving various AI models, including text generation, image generation, text-to-speech, text-to-video, and image-to-video models. It leverages Google Cloud Storage (GCS) for model storage and Hugging Face Hub for model downloads.

## Features

- **Unified API:** Single endpoint (`/generate`) for handling text generation tasks, with flexible parameters.
- **Model Loading from GCS:** Prioritizes loading models from Google Cloud Storage, which is faster and cost-efficient, and will automatically download and upload models from Hugging Face if not found in GCS.
- **Automatic Download to GCS:** If a model isn't found in GCS, it's automatically downloaded from Hugging Face Hub and uploaded to GCS for future use.
- **Text Generation:** Supports various text-generation models (e.g. GPT-2, Llama-3) using Hugging Face Transformers, including streaming responses.
- **Image Generation:** Supports Stable Diffusion for text-to-image generation.
- **Flux Image Generation:** Supports the Flux model for advanced text-to-image generation.
- **Text-to-Speech:** Utilizes AudioGen for text-to-speech synthesis.
- **Text-to-Video:** Basic text-to-video functionality using a pre-existing pipeline (requires a model loaded with pipeline tag "text-to-video").
- **Image-to-Video:** Supports image-to-video generation using CogVideo, including loading the input image from GCS.
- **Streaming Responses:** For text generation, supports streaming responses using `StreamingResponse`.
- **Error Handling:** Provides detailed error messages for debugging.
- **Environment Variable Configuration:** Uses environment variables to manage sensitive information like credentials and bucket names.

## Prerequisites

-   **Python 3.9+**
-   **Google Cloud Project:** A Google Cloud project with a GCS bucket for model storage.
-   **Hugging Face Account:** A Hugging Face account and an API token for downloading models from Hugging Face Hub.
-   **Environment Variables:** Ensure the following environment variables are set:
    -   `GCS_BUCKET_NAME`: The name of your Google Cloud Storage bucket.
    -   `GOOGLE_APPLICATION_CREDENTIALS_JSON`: The contents of your Google Cloud service account key in JSON format.
    -   `HF_API_TOKEN`: Your Hugging Face API token.

## Installation

1.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-directory>
    ```
2.  **Create a Virtual Environment (Optional but Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate  # On Windows
    ```
3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Ensure Environment Variables are Set:** Set `GCS_BUCKET_NAME`, `GOOGLE_APPLICATION_CREDENTIALS_JSON`, and `HF_API_TOKEN` either in your shell, or by exporting them, such as:
    ```bash
    export GCS_BUCKET_NAME="your-gcs-bucket-name"
    export GOOGLE_APPLICATION_CREDENTIALS_JSON='{"type": "service_account", ...}'
    export HF_API_TOKEN="your-hugging-face-api-token"
    ```
    *Note: Please make sure that the GOOGLE_APPLICATION_CREDENTIALS_JSON is a valid JSON String. You can obtain this by copying the contents of your credentials JSON file and pasting them as a single string.*

## Running the Application

```bash
python your_script_name.py
content_copy
Use code with caution.
Markdown

(Replace your_script_name.py with the name of your main Python script)

The API will be available at http://0.0.0.0:7860.

API Usage
/generate Endpoint

Handles all text generation tasks based on the task_type.

Request Body:

{
    "model_name": "gpt2",
    "input_text": "Hello, world!",
    "task_type": "text-to-text",
    "temperature": 1.0,
    "max_new_tokens": 200,
    "top_p": 1.0,
    "top_k": 50,
    "repetition_penalty": 1.0,
    "num_return_sequences": 1,
    "do_sample": true,
    "chunk_delay": 0.0,
    "stop_sequences": []
    "image_path": null # Optional, used only for image-to-video
}

Json

model_name: The Hugging Face model name (e.g., gpt2, meta-llama/Llama-3-8B-Instruct, stabilityai/stable-diffusion-2-1, facebook/audiogen-medium, THUDM/CogVideoX1.5-5B-I2V, black-forest-labs/FLUX.1-dev)

input_text: Input text for generation (prompt).

task_type: The type of task (text-to-text, text-to-image, text-to-speech, text-to-video, image-to-video, text-to-image-flux, text-generation-llama3).

temperature, max_new_tokens, top_p, top_k, repetition_penalty, num_return_sequences, do_sample, chunk_delay, stop_sequences: Generation configuration parameters (text-to-text generation related).

image_path: The GCS path to the image to be used in image-to-video (only used when task_type is image-to-video).

Example Text-to-Text Request:

curl -X POST -H "Content-Type: application/json" -d '{"model_name": "gpt2", "input_text": "Write a short story about a cat", "task_type": "text-to-text"}' http://localhost:7860/generate

Bash

Example Llama-3 Text Generation Request:

curl -X POST -H "Content-Type: application/json" -d '{"model_name": "meta-llama/Llama-3-8B-Instruct", "input_text": "What is the meaning of life?", "task_type": "text-generation-llama3", "max_new_tokens": 500}' http://localhost:7860/generate

Bash

Response: (Streaming for text-to-text, JSON for others)

{ "text": "Your text output..."}
content_copy
Use code with caution.
/generate-image Endpoint

Handles text-to-image generation using Stable Diffusion.

Request Body: Same as /generate with "task_type": "text-to-image".

Example Request:

curl -X POST -H "Content-Type: application/json" -d '{"model_name": "stabilityai/stable-diffusion-2-1", "input_text": "A cat in space", "task_type": "text-to-image"}' http://localhost:7860/generate-image

Bash

Response: Image in PNG format.

/generate-image-flux Endpoint

Handles text-to-image generation using Flux.

Request Body: Same as /generate with "task_type": "text-to-image-flux".

Example Request:

curl -X POST -H "Content-Type: application/json" -d '{"model_name": "black-forest-labs/FLUX.1-dev", "input_text": "A cat in space", "task_type": "text-to-image-flux"}' http://localhost:7860/generate-image-flux
content_copy
Use code with caution.
Bash

Response: Image in PNG format.

/generate-text-to-speech Endpoint

Handles text-to-speech synthesis using AudioGen.

Request Body: Same as /generate with "task_type": "text-to-speech".

Example Request:

curl -X POST -H "Content-Type: application/json" -d '{"model_name": "facebook/audiogen-medium", "input_text": "This is a test audio.", "task_type": "text-to-speech"}' http://localhost:7860/generate-text-to-speech

Bash

Response: Audio in WAV format.

/generate-video Endpoint

Handles basic text-to-video using a model loaded with pipeline tag "text-to-video".

Request Body: Same as /generate with "task_type": "text-to-video".

Example Request:

curl -X POST -H "Content-Type: application/json" -d '{"model_name": "model-with-text-to-video-pipeline", "input_text": "A car driving fast.", "task_type": "text-to-video"}' http://localhost:7860/generate-video

Bash

Response: Video in MP4 format.

/generate-image-to-video Endpoint

Handles image-to-video generation using CogVideo.

Request Body:

{
    "model_name": "THUDM/CogVideoX1.5-5B-I2V",
    "input_text": "A little girl is riding a bicycle at high speed.",
    "task_type": "image-to-video",
    "image_path": "path/to/your/image.jpg"
}
Json

Important: Please remember to replace path/to/your/image.jpg with the actual path to your image in GCS.

Example Request:

curl -X POST -H "Content-Type: application/json" -d '{"model_name": "THUDM/CogVideoX1.5-5B-I2V", "input_text": "A little girl is riding a bicycle at high speed.", "task_type": "image-to-video", "image_path": "path/to/your/image.jpg"}' http://localhost:7860/generate-image-to-video

Bash

Response: Video in MP4 format.

Important Notes

Model Loading Time: Loading models from GCS (or Hugging Face Hub) for the first time can take time, depending on model size. This might result in a longer initial request time.

Resource Utilization: Certain models and tasks (e.g. text-to-video, image-to-video) can be computationally demanding. Ensure you have adequate CPU resources, or add a GPU.

Model Compatibility: Make sure to use a model that is compatible with the selected task_type, otherwise an error will occur. For example use models with the pipeline tag "text-to-image" when task_type is set to "text-to-image".

GCS Image Paths: For image-to-video the image_path should point to an image in your GCS Bucket.

JSON Credentials: You must load the entire JSON object for your Google Cloud Service Account as a single string when using the GOOGLE_APPLICATION_CREDENTIALS_JSON environment variable.

Contributing

Contributions are welcome! If you find a bug or would like to suggest an enhancement, please open an issue or submit a pull request.

License

[Insert License Here]

**How to Use This `README.md`:**

1.  **Save:** Save the markdown text above into a file named `README.md` in your project's root directory.
2.  **View:** When you open this file in a Markdown viewer (like GitHub, GitLab, or VS Code), it will be rendered as a formatted document.

This `README.md` file should give anyone who visits your project a good overview of what it does, how to set it up, and how to use it. Make sure to replace the placeholders (e.g., `<your-repository-url>`) with your actual information.
