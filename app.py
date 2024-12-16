import os
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, field_validator
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    StoppingCriteriaList,
    pipeline as transformers_pipeline
)
from google.cloud import storage
from google.auth.exceptions import DefaultCredentialsError
import uvicorn
import asyncio
from io import BytesIO
import json
import logging
from diffusers import DiffusionPipeline, CogVideoXImageToVideoPipeline, FluxPipeline
from audiocraft.models import AudioGen
#from audiocraft.utils import convert_audio
from huggingface_hub import hf_hub_download, HfApi
from PIL import Image

GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
GOOGLE_APPLICATION_CREDENTIALS_JSON = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
HUGGINGFACE_HUB_TOKEN = os.getenv("HF_API_TOKEN")

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


try:
    credentials_info = json.loads(GOOGLE_APPLICATION_CREDENTIALS_JSON)
    client = storage.Client.from_service_account_info(credentials_info)
    bucket = client.get_bucket(GCS_BUCKET_NAME)
    logger.info(f"Conexión con Google Cloud Storage exitosa. Bucket: {GCS_BUCKET_NAME}")

except (DefaultCredentialsError, json.JSONDecodeError, KeyError, ValueError) as e:
    logger.error(f"Error al cargar las credenciales o bucket: {e}")
    raise RuntimeError(f"Error al cargar las credenciales o bucket: {e}")

app = FastAPI()

class GenerateRequest(BaseModel):
    model_name: str
    input_text: str = ""
    task_type: str
    temperature: float = 1.0
    max_new_tokens: int = 200
    top_p: float = 1.0
    top_k: int = 50
    repetition_penalty: float = 1.0
    num_return_sequences: int = 1
    do_sample: bool = True
    chunk_delay: float = 0.0
    stop_sequences: list[str] = []
    image_path: str = None # Added for image-to-video and similar cases

    @field_validator("model_name")
    def model_name_cannot_be_empty(cls, v):
        if not v:
            raise ValueError("model_name cannot be empty.")
        return v

    @field_validator("task_type")
    def task_type_must_be_valid(cls, v):
        valid_types = ["text-to-text", "text-to-image", "text-to-speech", "text-to-video", "image-to-video", "text-to-image-flux", "text-generation-llama3"]
        if v not in valid_types:
            raise ValueError(f"task_type must be one of: {valid_types}")
        return v

class GCSModelLoader:
    def __init__(self, bucket, client):
        self.bucket = bucket
        self.client = client
        self.hf_api = HfApi(token=HUGGINGFACE_HUB_TOKEN)

    def _get_gcs_blob_names(self, model_name):
        blobs = self.client.list_blobs(self.bucket, prefix=model_name)
        return [blob.name for blob in blobs]

    def _get_gcs_blob(self, blob_name):
        return self.bucket.blob(blob_name)

    def _get_gcs_file_like_object(self, blob_name):
        blob = self._get_gcs_blob(blob_name)
        return BytesIO(blob.download_as_bytes())

    async def load_model_and_tokenizer(self, model_name):
        gcs_blob_names = self._get_gcs_blob_names(model_name)
        if not gcs_blob_names:
             try:
                config = AutoConfig.from_pretrained(model_name)
                tokenizer = AutoTokenizer.from_pretrained(model_name, config=config)
                model = AutoModelForCausalLM.from_pretrained(model_name, config=config)

                if tokenizer.eos_token_id is not None and tokenizer.pad_token_id is None:
                    tokenizer.pad_token_id = config.pad_token_id or tokenizer.eos_token_id
                return model, tokenizer
             except Exception as e:
                try:
                   await self._download_and_upload_model_to_gcs(model_name)
                   config = AutoConfig.from_pretrained(model_name)
                   tokenizer = AutoTokenizer.from_pretrained(model_name, config=config)
                   model = AutoModelForCausalLM.from_pretrained(model_name, config=config)
                   if tokenizer.eos_token_id is not None and tokenizer.pad_token_id is None:
                        tokenizer.pad_token_id = config.pad_token_id or tokenizer.eos_token_id
                   return model, tokenizer
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"Error loading model from local or GCS: {e}")
        try:
            config_blob_name = next((name for name in gcs_blob_names if 'config.json' in name), None)
            if not config_blob_name:
                raise Exception(f"config.json not found in GCS for model {model_name}")
            config_file_obj = self._get_gcs_file_like_object(config_blob_name)
            config = AutoConfig.from_pretrained(model_name, local_files_only=True, cache_dir=None,  config=config_file_obj)
            
            model_blob_name = next((name for name in gcs_blob_names if 'pytorch_model.bin' in name), None)
            if not model_blob_name:
                raise Exception(f"pytorch_model.bin not found in GCS for model {model_name}")
            model_file_obj = self._get_gcs_file_like_object(model_blob_name)
            model = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=True, config=config, cache_dir=None, state_dict=torch.load(model_file_obj, map_location="cpu"))
            
            tokenizer_blob_name = next((name for name in gcs_blob_names if 'tokenizer.json' in name), None)
            if not tokenizer_blob_name:
                tokenizer_blob_name = next((name for name in gcs_blob_names if 'tokenizer_config.json' in name), None)
                if not tokenizer_blob_name:
                    raise Exception(f"tokenizer.json or tokenizer_config.json not found in GCS for model {model_name}")
            tokenizer_file_obj = self._get_gcs_file_like_object(tokenizer_blob_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True, config=config, cache_dir=None, tokenizer_config=tokenizer_file_obj)
            
            if tokenizer.eos_token_id is not None and tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = config.pad_token_id or tokenizer.eos_token_id
            
            del config_file_obj
            del model_file_obj
            del tokenizer_file_obj
            
            return model, tokenizer
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error loading model from GCS: {e}")

    async def load_diffusion_pipeline(self, model_name):
        gcs_blob_names = self._get_gcs_blob_names(model_name)
        if not gcs_blob_names:
            try:
                pipeline = DiffusionPipeline.from_pretrained(model_name)
                return pipeline
            except Exception as e:
                try:
                   await self._download_and_upload_model_to_gcs(model_name)
                   pipeline = DiffusionPipeline.from_pretrained(model_name)
                   return pipeline
                except Exception as e:
                   raise HTTPException(status_code=500, detail=f"Error loading diffusion pipeline from local or GCS: {e}")
        try:
            pipeline_components = {}
            for component_dir in ['scheduler', 'text_encoder', 'tokenizer', 'unet', 'vae']:
                component_prefix = f"{model_name}/{component_dir}"
                component_blob_names = [name for name in gcs_blob_names if name.startswith(component_prefix)]
                if not component_blob_names:
                    continue

                if component_dir == "tokenizer":
                    config_blob_name = next((name for name in component_blob_names if 'tokenizer.json' in name), None)
                    if not config_blob_name:
                        config_blob_name = next((name for name in component_blob_names if 'tokenizer_config.json' in name), None)
                        if not config_blob_name:
                            raise Exception(f"tokenizer.json or tokenizer_config.json not found in GCS for {component_dir}")
                    config_file_obj = self._get_gcs_file_like_object(config_blob_name)
                    pipeline_components[component_dir] = AutoTokenizer.from_pretrained(model_name, local_files_only=True, cache_dir=None, tokenizer_config=config_file_obj)
                    del config_file_obj
                elif component_dir == "scheduler":
                    config_blob_name = next((name for name in component_blob_names if 'scheduler_config.json' in name), None)
                    if not config_blob_name:
                        raise Exception(f"scheduler_config.json not found in GCS for {component_dir}")
                    config_file_obj = self._get_gcs_file_like_object(config_blob_name)
                    pipeline_components[component_dir] = AutoConfig.from_pretrained(model_name, local_files_only=True, cache_dir=None, config=config_file_obj)
                    del config_file_obj
                else:
                    config_blob_name = next((name for name in component_blob_names if 'config.json' in name), None)
                    if not config_blob_name:
                        raise Exception(f"config.json not found in GCS for {component_dir}")
                    config_file_obj = self._get_gcs_file_like_object(config_blob_name)
                    model_blob_name = next((name for name in component_blob_names if 'pytorch_model.bin' in name), None)
                    if not model_blob_name:
                        raise Exception(f"pytorch_model.bin not found in GCS for {component_dir}")
                    model_file_obj = self._get_gcs_file_like_object(model_blob_name)
                    pipeline_components[component_dir] = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=True, config=config_file_obj, cache_dir=None, state_dict=torch.load(model_file_obj, map_location="cpu"))
                    del config_file_obj
                    del model_file_obj

            pipeline = DiffusionPipeline(**pipeline_components)
            del pipeline_components
            return pipeline
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error loading diffusion pipeline from GCS: {e}")
    
    async def load_flux_pipeline(self, model_name):
        gcs_blob_names = self._get_gcs_blob_names(model_name)
        if not gcs_blob_names:
            try:
                pipeline = FluxPipeline.from_pretrained(model_name, torch_dtype=torch.bfloat16)
                return pipeline
            except Exception as e:
                try:
                   await self._download_and_upload_model_to_gcs(model_name)
                   pipeline = FluxPipeline.from_pretrained(model_name, torch_dtype=torch.bfloat16)
                   return pipeline
                except Exception as e:
                   raise HTTPException(status_code=500, detail=f"Error loading flux pipeline from local or GCS: {e}")
        try:
           
            pipeline_components = {}
            for component_dir in ['scheduler', 'text_encoder', 'tokenizer', 'unet', 'vae']:
                 component_prefix = f"{model_name}/{component_dir}"
                 component_blob_names = [name for name in gcs_blob_names if name.startswith(component_prefix)]
                 if not component_blob_names:
                    continue
                 if component_dir == "tokenizer":
                    config_blob_name = next((name for name in component_blob_names if 'tokenizer.json' in name), None)
                    if not config_blob_name:
                       config_blob_name = next((name for name in component_blob_names if 'tokenizer_config.json' in name), None)
                       if not config_blob_name:
                            raise Exception(f"tokenizer.json or tokenizer_config.json not found in GCS for {component_dir}")
                    config_file_obj = self._get_gcs_file_like_object(config_blob_name)
                    pipeline_components[component_dir] = AutoTokenizer.from_pretrained(model_name, local_files_only=True, cache_dir=None, tokenizer_config=config_file_obj)
                    del config_file_obj
                 elif component_dir == "scheduler":
                    config_blob_name = next((name for name in component_blob_names if 'scheduler_config.json' in name), None)
                    if not config_blob_name:
                        raise Exception(f"scheduler_config.json not found in GCS for {component_dir}")
                    config_file_obj = self._get_gcs_file_like_object(config_blob_name)
                    pipeline_components[component_dir] = AutoConfig.from_pretrained(model_name, local_files_only=True, cache_dir=None, config=config_file_obj)
                    del config_file_obj
                 else:
                    config_blob_name = next((name for name in component_blob_names if 'config.json' in name), None)
                    if not config_blob_name:
                        raise Exception(f"config.json not found in GCS for {component_dir}")
                    config_file_obj = self._get_gcs_file_like_object(config_blob_name)
                    model_blob_name = next((name for name in component_blob_names if 'pytorch_model.bin' in name), None)
                    if not model_blob_name:
                        raise Exception(f"pytorch_model.bin not found in GCS for {component_dir}")
                    model_file_obj = self._get_gcs_file_like_object(model_blob_name)
                    pipeline_components[component_dir] = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=True, config=config_file_obj, cache_dir=None, state_dict=torch.load(model_file_obj, map_location="cpu"))
                    del config_file_obj
                    del model_file_obj

            pipeline = FluxPipeline(**pipeline_components, torch_dtype=torch.bfloat16)
            del pipeline_components
            return pipeline
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error loading flux pipeline from GCS: {e}")

    async def load_cogvideo_pipeline(self, model_name):
            gcs_blob_names = self._get_gcs_blob_names(model_name)
            if not gcs_blob_names:
                try:
                    pipeline = CogVideoXImageToVideoPipeline.from_pretrained(model_name, torch_dtype=torch.bfloat16)
                    return pipeline
                except Exception as e:
                    try:
                      await self._download_and_upload_model_to_gcs(model_name)
                      pipeline = CogVideoXImageToVideoPipeline.from_pretrained(model_name, torch_dtype=torch.bfloat16)
                      return pipeline
                    except Exception as e:
                        raise HTTPException(status_code=500, detail=f"Error loading CogVideo pipeline from local or GCS: {e}")
            try:
                 pipeline_components = {}
                 for component_dir in ['scheduler', 'text_encoder', 'tokenizer', 'unet', 'vae']:
                     component_prefix = f"{model_name}/{component_dir}"
                     component_blob_names = [name for name in gcs_blob_names if name.startswith(component_prefix)]
                     if not component_blob_names:
                            continue
                     if component_dir == "tokenizer":
                        config_blob_name = next((name for name in component_blob_names if 'tokenizer.json' in name), None)
                        if not config_blob_name:
                            config_blob_name = next((name for name in component_blob_names if 'tokenizer_config.json' in name), None)
                            if not config_blob_name:
                                raise Exception(f"tokenizer.json or tokenizer_config.json not found in GCS for {component_dir}")
                        config_file_obj = self._get_gcs_file_like_object(config_blob_name)
                        pipeline_components[component_dir] = AutoTokenizer.from_pretrained(model_name, local_files_only=True, cache_dir=None, tokenizer_config=config_file_obj)
                        del config_file_obj
                     elif component_dir == "scheduler":
                        config_blob_name = next((name for name in component_blob_names if 'scheduler_config.json' in name), None)
                        if not config_blob_name:
                            raise Exception(f"scheduler_config.json not found in GCS for {component_dir}")
                        config_file_obj = self._get_gcs_file_like_object(config_blob_name)
                        pipeline_components[component_dir] = AutoConfig.from_pretrained(model_name, local_files_only=True, cache_dir=None, config=config_file_obj)
                        del config_file_obj
                     else:
                        config_blob_name = next((name for name in component_blob_names if 'config.json' in name), None)
                        if not config_blob_name:
                            raise Exception(f"config.json not found in GCS for {component_dir}")
                        config_file_obj = self._get_gcs_file_like_object(config_blob_name)
                        model_blob_name = next((name for name in component_blob_names if 'pytorch_model.bin' in name), None)
                        if not model_blob_name:
                           raise Exception(f"pytorch_model.bin not found in GCS for {component_dir}")
                        model_file_obj = self._get_gcs_file_like_object(model_blob_name)
                        pipeline_components[component_dir] = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=True, config=config_file_obj, cache_dir=None, state_dict=torch.load(model_file_obj, map_location="cpu"))
                        del config_file_obj
                        del model_file_obj

                 pipeline = CogVideoXImageToVideoPipeline(**pipeline_components, torch_dtype=torch.bfloat16)
                 del pipeline_components
                 return pipeline
            except Exception as e:
              raise HTTPException(status_code=500, detail=f"Error loading CogVideo pipeline from GCS: {e}")

    async def load_audiogen_model(self, model_name):
        gcs_blob_names = self._get_gcs_blob_names(model_name)
        if not gcs_blob_names:
            try:
              model = AudioGen.get_pretrained(model_name, device='cpu')
              return model
            except Exception as e:
              try:
                 await self._download_and_upload_model_to_gcs(model_name)
                 model = AudioGen.get_pretrained(model_name, device='cpu')
                 return model
              except Exception as e:
                  raise HTTPException(status_code=500, detail=f"Error loading AudioGen model from local or GCS: {e}")
        try:
            config_blob_name = next((name for name in gcs_blob_names if 'config.json' in name), None)
            if not config_blob_name:
                raise Exception(f"config.json not found in GCS for AudioGen model {model_name}")
            config_file_obj = self._get_gcs_file_like_object(config_blob_name)
            state_dict_blob_name = next((name for name in gcs_blob_names if 'model.pt' in name), None)
            if not state_dict_blob_name:
                raise Exception(f"model.pt not found in GCS for AudioGen model {model_name}")
            state_dict_file_obj = self._get_gcs_file_like_object(state_dict_blob_name)
            model = AudioGen.get_pretrained(model_name, device='cpu', state_dict=torch.load(state_dict_file_obj, map_location="cpu"), config=config_file_obj)
            del config_file_obj
            del state_dict_file_obj

            return model
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error loading AudioGen model from GCS: {e}")
    
    async def _load_text_generation_pipeline(self, model_name):
        gcs_blob_names = self._get_gcs_blob_names(model_name)
        if not gcs_blob_names:
            try:
                pipeline = transformers_pipeline(
                "text-generation",
                model=model_name,
                model_kwargs={"torch_dtype": torch.bfloat16},
                device_map="auto",
                )
                return pipeline
            except Exception as e:
                try:
                   await self._download_and_upload_model_to_gcs(model_name)
                   pipeline = transformers_pipeline(
                        "text-generation",
                        model=model_name,
                        model_kwargs={"torch_dtype": torch.bfloat16},
                        device_map="auto",
                    )
                   return pipeline
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"Error loading text generation pipeline from local or GCS: {e}")
        try:
             config_blob_name = next((name for name in gcs_blob_names if 'config.json' in name), None)
             if not config_blob_name:
                 raise Exception(f"config.json not found in GCS for Llama-3 model {model_name}")
             config_file_obj = self._get_gcs_file_like_object(config_blob_name)
             
             model_blob_name = next((name for name in gcs_blob_names if 'pytorch_model.bin' in name), None)
             if not model_blob_name:
                  raise Exception(f"pytorch_model.bin not found in GCS for Llama-3 model {model_name}")
             model_file_obj = self._get_gcs_file_like_object(model_blob_name)

             tokenizer_blob_name = next((name for name in gcs_blob_names if 'tokenizer.json' in name), None)
             if not tokenizer_blob_name:
                tokenizer_blob_name = next((name for name in gcs_blob_names if 'tokenizer_config.json' in name), None)
                if not tokenizer_blob_name:
                    raise Exception(f"tokenizer.json or tokenizer_config.json not found in GCS for model {model_name}")
             tokenizer_file_obj = self._get_gcs_file_like_object(tokenizer_blob_name)
            
             pipeline = transformers_pipeline(
                "text-generation",
                model=model_name,
                model_kwargs={"torch_dtype": torch.bfloat16, "config":config_file_obj , "state_dict":torch.load(model_file_obj, map_location="cpu")},
                tokenizer=AutoTokenizer.from_pretrained(model_name, local_files_only=True, cache_dir=None, tokenizer_config=tokenizer_file_obj),
                device_map="auto",
            )
             
             del config_file_obj
             del model_file_obj
             del tokenizer_file_obj
            
             return pipeline
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error loading Llama-3 model from GCS: {e}")
        
    async def _download_and_upload_model_to_gcs(self, model_name):
        logger.info(f"Downloading and uploading model: {model_name} from Hugging Face Hub to GCS")
        
        model_info = self.hf_api.model_info(model_name)
        
        if model_info.pipeline_tag == "text-generation":
            await self._download_and_upload_text_generation_model(model_name)
        elif model_info.pipeline_tag == "text-to-image":
             await self._download_and_upload_diffusion_model(model_name)
        elif "audio" in model_info.pipeline_tag:
            await self._download_and_upload_audio_model(model_name)
        elif "image-to-video" in model_info.pipeline_tag:
            await self._download_and_upload_video_model(model_name)
        else:
           raise Exception(f"Model type {model_info.pipeline_tag} not supported for automatic download and upload.")

        logger.info(f"Finished downloading and uploading model {model_name} to GCS")
    
    async def _download_and_upload_text_generation_model(self, model_name):
          
          config_file = hf_hub_download(model_name, filename="config.json", token=HUGGINGFACE_HUB_TOKEN)
          self._upload_file_to_gcs(config_file, os.path.join(model_name, "config.json"))
          
          model_file = hf_hub_download(model_name, filename="pytorch_model.bin", token=HUGGINGFACE_HUB_TOKEN)
          self._upload_file_to_gcs(model_file, os.path.join(model_name, "pytorch_model.bin"))

          try:
               tokenizer_file = hf_hub_download(model_name, filename="tokenizer.json", token=HUGGINGFACE_HUB_TOKEN)
               self._upload_file_to_gcs(tokenizer_file, os.path.join(model_name, "tokenizer.json"))
          except:
              tokenizer_file = hf_hub_download(model_name, filename="tokenizer_config.json", token=HUGGINGFACE_HUB_TOKEN)
              self._upload_file_to_gcs(tokenizer_file, os.path.join(model_name, "tokenizer_config.json"))

    async def _download_and_upload_diffusion_model(self, model_name):
         
          subfolders = ["scheduler", "text_encoder", "tokenizer", "unet", "vae"]

          for subfolder in subfolders:
             try:
                config_file = hf_hub_download(model_name, filename=os.path.join(subfolder, "config.json"), token=HUGGINGFACE_HUB_TOKEN)
                self._upload_file_to_gcs(config_file, os.path.join(model_name, subfolder, "config.json"))
             except:
                 try:
                    config_file = hf_hub_download(model_name, filename=os.path.join(subfolder, "scheduler_config.json"), token=HUGGINGFACE_HUB_TOKEN)
                    self._upload_file_to_gcs(config_file, os.path.join(model_name, subfolder, "scheduler_config.json"))
                 except:
                     try:
                        tokenizer_file = hf_hub_download(model_name, filename=os.path.join(subfolder, "tokenizer.json"), token=HUGGINGFACE_HUB_TOKEN)
                        self._upload_file_to_gcs(tokenizer_file, os.path.join(model_name, subfolder, "tokenizer.json"))
                     except:
                         try:
                            tokenizer_file = hf_hub_download(model_name, filename=os.path.join(subfolder, "tokenizer_config.json"), token=HUGGINGFACE_HUB_TOKEN)
                            self._upload_file_to_gcs(tokenizer_file, os.path.join(model_name, subfolder, "tokenizer_config.json"))
                         except:
                             pass

             try:
                model_file = hf_hub_download(model_name, filename=os.path.join(subfolder, "pytorch_model.bin"), token=HUGGINGFACE_HUB_TOKEN)
                self._upload_file_to_gcs(model_file, os.path.join(model_name, subfolder, "pytorch_model.bin"))
             except:
                 pass

    async def _download_and_upload_audio_model(self, model_name):

        config_file = hf_hub_download(model_name, filename="config.json", token=HUGGINGFACE_HUB_TOKEN)
        self._upload_file_to_gcs(config_file, os.path.join(model_name, "config.json"))
        
        model_file = hf_hub_download(model_name, filename="model.pt", token=HUGGINGFACE_HUB_TOKEN)
        self._upload_file_to_gcs(model_file, os.path.join(model_name, "model.pt"))

    async def _download_and_upload_video_model(self, model_name):
         
          subfolders = ["scheduler", "text_encoder", "tokenizer", "unet", "vae"]

          for subfolder in subfolders:
             try:
                config_file = hf_hub_download(model_name, filename=os.path.join(subfolder, "config.json"), token=HUGGINGFACE_HUB_TOKEN)
                self._upload_file_to_gcs(config_file, os.path.join(model_name, subfolder, "config.json"))
             except:
                 try:
                    config_file = hf_hub_download(model_name, filename=os.path.join(subfolder, "scheduler_config.json"), token=HUGGINGFACE_HUB_TOKEN)
                    self._upload_file_to_gcs(config_file, os.path.join(model_name, subfolder, "scheduler_config.json"))
                 except:
                     try:
                        tokenizer_file = hf_hub_download(model_name, filename=os.path.join(subfolder, "tokenizer.json"), token=HUGGINGFACE_HUB_TOKEN)
                        self._upload_file_to_gcs(tokenizer_file, os.path.join(model_name, subfolder, "tokenizer.json"))
                     except:
                         try:
                            tokenizer_file = hf_hub_download(model_name, filename=os.path.join(subfolder, "tokenizer_config.json"), token=HUGGINGFACE_HUB_TOKEN)
                            self._upload_file_to_gcs(tokenizer_file, os.path.join(model_name, subfolder, "tokenizer_config.json"))
                         except:
                             pass

             try:
                model_file = hf_hub_download(model_name, filename=os.path.join(subfolder, "pytorch_model.bin"), token=HUGGINGFACE_HUB_TOKEN)
                self._upload_file_to_gcs(model_file, os.path.join(model_name, subfolder, "pytorch_model.bin"))
             except:
                 pass
    
    def _upload_file_to_gcs(self, file_path, gcs_path):
        blob = self.bucket.blob(gcs_path)
        blob.upload_from_filename(file_path)
        os.remove(file_path)


model_loader = GCSModelLoader(bucket, client)

@app.post("/generate")
async def generate(request: GenerateRequest):
    try:
        model_name = request.model_name
        input_text = request.input_text
        task_type = request.task_type
        temperature = request.temperature
        max_new_tokens = request.max_new_tokens
        top_p = request.top_p
        top_k = request.top_k
        repetition_penalty = request.repetition_penalty
        num_return_sequences = request.num_return_sequences
        do_sample = request.do_sample
        chunk_delay = request.chunk_delay
        stop_sequences = request.stop_sequences
        image_path = request.image_path

        if task_type == "text-to-text":
             model, tokenizer = await model_loader.load_model_and_tokenizer(model_name)
             model.to("cpu")
             generation_config = GenerationConfig(
                 temperature=temperature,
                 max_new_tokens=max_new_tokens,
                 top_p=top_p,
                 top_k=top_k,
                 repetition_penalty=repetition_penalty,
                 do_sample=do_sample,
                 num_return_sequences=num_return_sequences,
             )
             response = StreamingResponse(
                 stream_text(model, tokenizer, input_text, generation_config, stop_sequences, "cpu", chunk_delay),
                 media_type="text/plain"
             )
             del model
             del tokenizer
             return response
        elif task_type == "text-generation-llama3":
            pipeline = await model_loader._load_text_generation_pipeline(model_name)
            pipeline.model.to("cpu")
            messages = [
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {"role": "user", "content": input_text},
                ]

            outputs = pipeline(
                    messages,
                    max_new_tokens=max_new_tokens,
                )
            del pipeline
            return { "text": outputs[0]["generated_text"][-1] }
        else:
            model, tokenizer = await model_loader.load_model_and_tokenizer(model_name)
            model.to("cpu")
            generation_config = GenerationConfig(
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                num_return_sequences=num_return_sequences,
            )
            all_text = ""
            async for chunk in stream_text(model, tokenizer, input_text, generation_config, stop_sequences, "cpu", chunk_delay):
                all_text += chunk
            
            del model
            del tokenizer
            return { "text": all_text }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

async def stream_text(model, tokenizer, input_text, generation_config, stop_sequences, device, chunk_delay, max_length=2048):
    with torch.no_grad():
        encoded_input = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
        input_length = encoded_input["input_ids"].shape[1]
        remaining_tokens = max_length - input_length
    
        if remaining_tokens <= 0:
            yield ""
            return
    
        generation_config.max_new_tokens = min(remaining_tokens, generation_config.max_new_tokens)
        
        def stop_criteria(input_ids, scores):
            decoded_output = tokenizer.decode(int(input_ids[0][-1]), skip_special_tokens=True)
            return decoded_output in stop_sequences
    
        stopping_criteria = StoppingCriteriaList([stop_criteria])
    
        output_ids = model.generate(
            **encoded_input,
            do_sample=generation_config.do_sample,
            max_new_tokens=generation_config.max_new_tokens,
            temperature=generation_config.temperature,
            top_p=generation_config.top_p,
            top_k=generation_config.top_k,
            repetition_penalty=generation_config.repetition_penalty,
            num_return_sequences=generation_config.num_return_sequences,
            stopping_criteria=stopping_criteria,
            output_scores=True,
            return_dict_in_generate=True,
            streamer=None
        )
    
    
        for output in output_ids.sequences:
            for token_id in output:
              token = tokenizer.decode(token_id, skip_special_tokens=True)
              yield token
              await asyncio.sleep(chunk_delay)
    del encoded_input
    del output_ids
    


@app.post("/generate-image")
async def generate_image(request: GenerateRequest):
    try:
        validated_body = request
        pipeline = await model_loader.load_diffusion_pipeline(validated_body.model_name)
        pipeline.to("cpu")
        with torch.no_grad():
            image = pipeline(validated_body.input_text).images[0]

        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format="PNG")
        img_byte_arr.seek(0)
        
        del pipeline
        del image

        return StreamingResponse(img_byte_arr, media_type="image/png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/generate-image-flux")
async def generate_image_flux(request: GenerateRequest):
    try:
        validated_body = request
        pipeline = await model_loader.load_flux_pipeline(validated_body.model_name)
        pipeline.enable_model_cpu_offload()
        with torch.no_grad():
             image = pipeline(
                validated_body.input_text,
                height=1024,
                width=1024,
                guidance_scale=3.5,
                num_inference_steps=50,
                max_sequence_length=512,
                generator=torch.Generator("cpu").manual_seed(0)
                ).images[0]

        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format="PNG")
        img_byte_arr.seek(0)
        del pipeline
        del image
        return StreamingResponse(img_byte_arr, media_type="image/png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/generate-text-to-speech")
async def generate_text_to_speech(request: GenerateRequest):
    try:
        validated_body = request
        
        model = await model_loader.load_audiogen_model(validated_body.model_name)
        model.to("cpu")
        
        descriptions = [validated_body.input_text]
        
        with torch.no_grad():
             output = model.generate(descriptions)
       
        audio_byte_arr = BytesIO()
        wav = convert_audio(output.cpu(), 32000, target_sr=16000)
        
        # Use torch_wav_to_bytes to handle torch tensor conversion to bytes
        def torch_wav_to_bytes(wav):
            with BytesIO() as byte_io:
                torch.save(wav, byte_io)
                byte_io.seek(0)
                return byte_io.read()
        
        audio_byte_arr.write(torch_wav_to_bytes(wav))
        
        audio_byte_arr.seek(0)
        del model
        del output
        del wav

        return StreamingResponse(audio_byte_arr, media_type="audio/wav")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/generate-video")
async def generate_video(request: GenerateRequest):
    try:
        validated_body = request
        video_generator = pipeline("text-to-video", model=validated_body.model_name, device="cpu")
        with torch.no_grad():
            video = video_generator(validated_body.input_text)[0]
        
        video_byte_arr = BytesIO()
        video.save(video_byte_arr)
        video_byte_arr.seek(0)
        del video_generator
        del video
        return StreamingResponse(video_byte_arr, media_type="video/mp4")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/generate-image-to-video")
async def generate_image_to_video(request: GenerateRequest):
    try:
        validated_body = request
        pipeline = await model_loader.load_cogvideo_pipeline(validated_body.model_name)
        pipeline.enable_sequential_cpu_offload()
        pipeline.vae.enable_tiling()
        pipeline.vae.enable_slicing()
        
        if not validated_body.image_path:
            raise HTTPException(status_code=400, detail="Image path is required for image-to-video task.")
        
        # Download image from GCS
        image_blob = bucket.blob(validated_body.image_path)
        image_bytes = image_blob.download_as_bytes()
        image = Image.open(BytesIO(image_bytes))
        
        
        with torch.no_grad():
             video = pipeline(
                prompt=validated_body.input_text,
                image=image,
                num_videos_per_prompt=1,
                num_inference_steps=50,
                num_frames=81,
                guidance_scale=6,
                generator=torch.Generator(device="cpu").manual_seed(42),
             ).frames[0]

        video_byte_arr = BytesIO()
        from diffusers.utils import export_to_video
        export_to_video(video, video_byte_arr, fps=8)
        video_byte_arr.seek(0)
        
        del pipeline
        del video
        del image
        return StreamingResponse(video_byte_arr, media_type="video/mp4")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
