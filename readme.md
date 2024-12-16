```markdown
# ✨ AI Model API Service ✨

A powerful FastAPI-based API for serving diverse AI models including text generation, image generation, text-to-speech, text-to-video, and image-to-video. Leverages Google Cloud Storage (GCS) for storage and Hugging Face Hub for downloads.

---

## 🚀 Features

-   **Unified API:** A single `/generate` endpoint for all generation tasks.
-   **GCS Prioritization:** Loads models from GCS for speed and cost, falling back to Hugging Face.
-   **Automatic Upload to GCS:** If a model is downloaded from Hugging Face, it's automatically uploaded to GCS.
-   **Text Generation:** Uses Hugging Face Transformers for diverse text generation models (including streaming responses).
-   **Image Generation:** Supports Stable Diffusion and Flux for text-to-image.
-   **Text-to-Speech:** Utilizes AudioGen for text-to-speech.
-   **Text-to-Video:** Provides basic text-to-video functionality.
-   **Image-to-Video:** Supports image-to-video generation using CogVideo.
-   **Streaming Responses:** Supports `StreamingResponse` for text generation.
-   **Environment Variable Configuration:** Uses environment variables for sensitive information.

---

## ⚙️ Prerequisites

Before you begin, make sure you have the following:

*   **Python 3.9+**
*   A Google Cloud Project with a GCS bucket (`GCS_BUCKET_NAME`)
*   A Google Cloud Service Account Key (`GOOGLE_APPLICATION_CREDENTIALS_JSON`)
*   A Hugging Face Account API Token (`HF_API_TOKEN`)

---

## ⬇️ Installation

1.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-directory>
    ```
2.  **Create a Virtual Environment (Optional):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate   # On Windows
    ```
3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Set Environment Variables:** You must set the `GCS_BUCKET_NAME`, `GOOGLE_APPLICATION_CREDENTIALS_JSON` (as a single JSON string), and `HF_API_TOKEN` environment variables. For example:
    ```bash
    export GCS_BUCKET_NAME="your-gcs-bucket-name"
    export GOOGLE_APPLICATION_CREDENTIALS_JSON='{"type": "service_account", ...}'
    export HF_API_TOKEN="your-hugging-face-api-token"
    ```
    ⚠️ **Important:** Please ensure `GOOGLE_APPLICATION_CREDENTIALS_JSON` is a valid JSON string.

---

## 🚀 Usage

### 💻 Running the Application

```bash
python your_script_name.py
```

The API will be available at `http://0.0.0.0:7860`.

### ⚙️ /generate Endpoint

Handles all generation tasks.

**Request Body:**

```json
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
    "stop_sequences": [],
    "image_path": null
}
```

**Parameters:**

-   `model_name`: (string) The Hugging Face model name (e.g., `gpt2`).
-   `input_text`: (string) Input text for generation.
-   `task_type`: (string) Type of task (`text-to-text`, `text-to-image`, `text-to-speech`, `text-to-video`, `image-to-video`, `text-to-image-flux`, `text-generation-llama3`).
-   `temperature`, `max_new_tokens`, etc.: (numbers, booleans) Generation configuration parameters (text-to-text related).
-   `image_path`:(string, optional) GCS path to an image, only used when `task_type` is `image-to-video`.

**Examples:**

*   **Text-to-Text:**
    ```bash
    curl -X POST -H "Content-Type: application/json" -d '{"model_name": "gpt2", "input_text": "Write a short story about a cat", "task_type": "text-to-text"}' http://localhost:7860/generate
    ```
*   **Llama-3 Text Generation:**
    ```bash
    curl -X POST -H "Content-Type: application/json" -d '{"model_name": "meta-llama/Llama-3-8B-Instruct", "input_text": "What is the meaning of life?", "task_type": "text-generation-llama3", "max_new_tokens": 500}' http://localhost:7860/generate
    ```

*   **Response (Text-to-Text/Llama-3):**
    ```json
    { "text": "Your text output..." }
    ```

### 🖼️ /generate-image Endpoint

Handles text-to-image generation (Stable Diffusion).

**Request Body:** Same as `/generate` with `"task_type": "text-to-image"`.

**Example:**
```bash
curl -X POST -H "Content-Type: application/json" -d '{"model_name": "stabilityai/stable-diffusion-2-1", "input_text": "A cat in space", "task_type": "text-to-image"}' http://localhost:7860/generate
```

**Response:** Image in PNG format.

### 🖼️ /generate-image-flux Endpoint

Handles text-to-image generation using Flux.

**Request Body:** Same as `/generate` with `"task_type": "text-to-image-flux"`.

**Example:**
```bash
curl -X POST -H "Content-Type: application/json" -d '{"model_name": "black-forest-labs/FLUX.1-dev", "input_text": "A cat in space", "task_type": "text-to-image-flux"}' http://localhost:7860/generate
```

**Response:** Image in PNG format.

### 🗣️ /generate-text-to-speech Endpoint

Handles text-to-speech synthesis using AudioGen.

**Request Body:** Same as `/generate` with `"task_type": "text-to-speech"`.

**Example:**
```bash
curl -X POST -H "Content-Type: application/json" -d '{"model_name": "facebook/audiogen-medium", "input_text": "This is a test audio.", "task_type": "text-to-speech"}' http://localhost:7860/generate
```

**Response:** Audio in WAV format.

### 🎬 /generate-video Endpoint

Handles basic text-to-video using a model loaded with pipeline tag "text-to-video".

**Request Body:** Same as `/generate` with `"task_type": "text-to-video"`.

**Example:**
```bash
curl -X POST -H "Content-Type: application/json" -d '{"model_name": "model-with-text-to-video-pipeline", "input_text": "A car driving fast.", "task_type": "text-to-video"}' http://localhost:7860/generate
```

**Response:** Video in MP4 format.

### 🎥 /generate-image-to-video Endpoint

Handles image-to-video generation using CogVideo.

**Request Body:**

```json
{
    "model_name": "THUDM/CogVideoX1.5-5B-I2V",
    "input_text": "A little girl is riding a bicycle at high speed.",
    "task_type": "image-to-video",
    "image_path": "path/to/your/image.jpg"
}
```

**Important:** Please remember to replace `path/to/your/image.jpg` with the actual path to your image in GCS.

**Example:**
```bash
curl -X POST -H "Content-Type: application/json" -d '{"model_name": "THUDM/CogVideoX1.5-5B-I2V", "input_text": "A little girl is riding a bicycle at high speed.", "task_type": "image-to-video", "image_path": "path/to/your/image.jpg"}' http://localhost:7860/generate
```

**Response:** Video in MP4 format.

---

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request.

---

## 📜 License

[Insert License Here]
```
