//! CLAP model wrapper for ONNX Runtime inference.

use std::path::Path;
use std::sync::{Arc, Mutex};

use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use tokenizers::Tokenizer;
use tracing::{debug, info, warn};

use super::{AudioData, AudioProcessor, Embedding, InferenceError, MelFeatures, ModelPaths, EMBEDDING_DIM};

/// Maximum sequence length for CLAP text models
const MAX_SEQ_LENGTH: usize = 77;

/// Device type for inference
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    Cpu,
    Cuda,
    Rocm,
    CoreML,
    DirectML,
    OpenVINO,
}

impl std::fmt::Display for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cpu => write!(f, "CPU"),
            Self::Cuda => write!(f, "CUDA"),
            Self::Rocm => write!(f, "ROCm"),
            Self::CoreML => write!(f, "CoreML"),
            Self::DirectML => write!(f, "DirectML"),
            Self::OpenVINO => write!(f, "OpenVINO"),
        }
    }
}

/// Configuration for device/accelerator selection
#[derive(Debug, Clone, Copy, Default)]
pub struct DeviceConfig {
    pub cuda: bool,
    pub rocm: bool,
    pub coreml: bool,
    pub directml: bool,
    pub openvino: bool,
}

impl DeviceConfig {
    /// Create config with CUDA enabled (backwards compatibility)
    pub fn with_cuda(use_cuda: bool) -> Self {
        Self {
            cuda: use_cuda,
            ..Default::default()
        }
    }

    /// Determine which device will be used based on config and available features
    pub fn selected_device(&self) -> Device {
        if self.cuda {
            #[cfg(feature = "cuda")]
            return Device::Cuda;
        }
        if self.rocm {
            #[cfg(feature = "rocm")]
            return Device::Rocm;
        }
        if self.coreml {
            #[cfg(feature = "coreml")]
            return Device::CoreML;
        }
        if self.directml {
            #[cfg(feature = "directml")]
            return Device::DirectML;
        }
        if self.openvino {
            #[cfg(feature = "openvino")]
            return Device::OpenVINO;
        }
        Device::Cpu
    }
}

/// CLAP model for generating text and audio embeddings
pub struct ClapModel {
    text_session: Mutex<Session>,
    audio_session: Mutex<Session>,
    audio_processor: Mutex<AudioProcessor>,
    tokenizer: Option<Tokenizer>,
    device: Device,
}

impl std::fmt::Debug for ClapModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ClapModel")
            .field("device", &self.device)
            .field("embedding_dim", &EMBEDDING_DIM)
            .field("has_tokenizer", &self.tokenizer.is_some())
            .finish()
    }
}

impl ClapModel {
    /// Load CLAP model from downloaded paths
    pub fn load(paths: &ModelPaths, use_cuda: bool) -> Result<Self, InferenceError> {
        Self::load_with_config(paths, DeviceConfig::with_cuda(use_cuda))
    }

    /// Load CLAP model with full device configuration
    pub fn load_with_config(
        paths: &ModelPaths,
        device_config: DeviceConfig,
    ) -> Result<Self, InferenceError> {
        let device = device_config.selected_device();

        info!(?device, "Loading CLAP model");

        let text_session = Self::create_session(&paths.text_model, &device_config)?;
        let audio_session = Self::create_session(&paths.audio_model, &device_config)?;

        // Log model info
        debug!(
            text_inputs = ?text_session.inputs.iter().map(|i| &i.name).collect::<Vec<_>>(),
            text_outputs = ?text_session.outputs.iter().map(|o| &o.name).collect::<Vec<_>>(),
            "Text model loaded"
        );
        debug!(
            audio_inputs = ?audio_session.inputs.iter().map(|i| &i.name).collect::<Vec<_>>(),
            audio_outputs = ?audio_session.outputs.iter().map(|o| &o.name).collect::<Vec<_>>(),
            "Audio model loaded"
        );

        // Load tokenizer if available
        let tokenizer = if let Some(ref tokenizer_path) = paths.tokenizer_config {
            match Tokenizer::from_file(tokenizer_path) {
                Ok(tok) => {
                    info!(?tokenizer_path, "Tokenizer loaded");
                    Some(tok)
                }
                Err(e) => {
                    warn!(?tokenizer_path, error = %e, "Failed to load tokenizer, using fallback");
                    None
                }
            }
        } else {
            warn!("No tokenizer path provided, using fallback tokenization");
            None
        };

        let audio_processor = AudioProcessor::new(48_000, 10.0, 10.0);

        Ok(Self {
            text_session: Mutex::new(text_session),
            audio_session: Mutex::new(audio_session),
            audio_processor: Mutex::new(audio_processor),
            tokenizer,
            device,
        })
    }

    fn create_session(
        model_path: &Path,
        device_config: &DeviceConfig,
    ) -> Result<Session, InferenceError> {
        // Read model bytes from file
        let model_bytes = std::fs::read(model_path)
            .map_err(|e| InferenceError::Onnx(format!("Failed to read model file: {e}")))?;

        let mut builder = Session::builder().map_err(|e| InferenceError::Onnx(e.to_string()))?;

        builder = builder
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| InferenceError::Onnx(e.to_string()))?;

        builder = builder
            .with_intra_threads(4)
            .map_err(|e| InferenceError::Onnx(e.to_string()))?;

        // Configure execution provider based on device config
        // Priority: CUDA > ROCm > CoreML > DirectML > OpenVINO > CPU
        if device_config.cuda {
            #[cfg(feature = "cuda")]
            {
                use ort::execution_providers::CUDAExecutionProvider;
                builder = builder
                    .with_execution_providers([CUDAExecutionProvider::default().build()])
                    .map_err(|e| InferenceError::Onnx(e.to_string()))?;
            }
            #[cfg(not(feature = "cuda"))]
            {
                warn!("CUDA requested but not compiled with cuda feature, using CPU");
            }
        } else if device_config.rocm {
            #[cfg(feature = "rocm")]
            {
                use ort::execution_providers::ROCmExecutionProvider;
                builder = builder
                    .with_execution_providers([ROCmExecutionProvider::default().build()])
                    .map_err(|e| InferenceError::Onnx(e.to_string()))?;
            }
            #[cfg(not(feature = "rocm"))]
            {
                warn!("ROCm requested but not compiled with rocm feature, using CPU");
            }
        } else if device_config.coreml {
            #[cfg(feature = "coreml")]
            {
                use ort::execution_providers::CoreMLExecutionProvider;
                builder = builder
                    .with_execution_providers([CoreMLExecutionProvider::default().build()])
                    .map_err(|e| InferenceError::Onnx(e.to_string()))?;
            }
            #[cfg(not(feature = "coreml"))]
            {
                warn!("CoreML requested but not compiled with coreml feature, using CPU");
            }
        } else if device_config.directml {
            #[cfg(feature = "directml")]
            {
                use ort::execution_providers::DirectMLExecutionProvider;
                builder = builder
                    .with_execution_providers([DirectMLExecutionProvider::default().build()])
                    .map_err(|e| InferenceError::Onnx(e.to_string()))?;
            }
            #[cfg(not(feature = "directml"))]
            {
                warn!("DirectML requested but not compiled with directml feature, using CPU");
            }
        } else if device_config.openvino {
            #[cfg(feature = "openvino")]
            {
                use ort::execution_providers::OpenVINOExecutionProvider;
                builder = builder
                    .with_execution_providers([OpenVINOExecutionProvider::default().build()])
                    .map_err(|e| InferenceError::Onnx(e.to_string()))?;
            }
            #[cfg(not(feature = "openvino"))]
            {
                warn!("OpenVINO requested but not compiled with openvino feature, using CPU");
            }
        }

        builder
            .commit_from_memory(&model_bytes)
            .map_err(|e| InferenceError::Onnx(format!("Failed to load model: {e}")))
    }

    /// Get the device being used for inference
    pub fn device(&self) -> Device {
        self.device
    }

    /// Generate text embedding from input text
    pub fn text_embedding(&self, text: &str) -> Result<Embedding, InferenceError> {
        let outputs = self.run_text_inference(text)?;
        let mut embedding = Embedding::from_raw(outputs);
        embedding.normalize();
        Ok(embedding)
    }

    /// Tokenize text using the loaded tokenizer or fallback
    fn tokenize_text(&self, text: &str) -> Result<(Vec<i64>, Vec<i64>), InferenceError> {
        if let Some(ref tokenizer) = self.tokenizer {
            // Use the real tokenizer
            let encoding = tokenizer
                .encode(text, true)
                .map_err(|e| InferenceError::Onnx(format!("Tokenization failed: {e}")))?;

            let ids = encoding.get_ids();
            let attention = encoding.get_attention_mask();

            // Truncate or pad to MAX_SEQ_LENGTH
            let mut input_ids = vec![0i64; MAX_SEQ_LENGTH];
            let mut attention_mask = vec![0i64; MAX_SEQ_LENGTH];

            let len = ids.len().min(MAX_SEQ_LENGTH);
            for i in 0..len {
                input_ids[i] = ids[i] as i64;
                attention_mask[i] = attention[i] as i64;
            }

            debug!(
                text_len = text.len(),
                token_count = ids.len(),
                truncated_to = len,
                "Text tokenized"
            );

            Ok((input_ids, attention_mask))
        } else {
            // Fallback: simple word-based tokenization (not recommended for production)
            // This produces placeholder IDs that won't give meaningful embeddings
            warn!("Using fallback tokenization - embeddings may not be meaningful");

            let tokens: Vec<i64> = text
                .split_whitespace()
                .take(MAX_SEQ_LENGTH - 2) // Leave room for special tokens
                .enumerate()
                .map(|(i, _)| i as i64 + 1)
                .collect();

            let mut input_ids = vec![0i64; MAX_SEQ_LENGTH];
            let mut attention_mask = vec![0i64; MAX_SEQ_LENGTH];

            // Add CLS token at start (typically ID 101 for BERT-like models)
            input_ids[0] = 101;
            attention_mask[0] = 1;

            // Add word tokens
            for (i, &t) in tokens.iter().enumerate() {
                input_ids[i + 1] = t;
                attention_mask[i + 1] = 1;
            }

            // Add SEP token at end (typically ID 102 for BERT-like models)
            input_ids[tokens.len() + 1] = 102;
            attention_mask[tokens.len() + 1] = 1;

            Ok((input_ids, attention_mask))
        }
    }

    fn run_text_inference(&self, text: &str) -> Result<Vec<f32>, InferenceError> {
        // Tokenize the input text
        let (input_ids_data, attention_mask_data) = self.tokenize_text(text)?;

        // Create tensors using Tensor::from_array with shape tuple
        let input_ids =
            Tensor::from_array(([1usize, MAX_SEQ_LENGTH], input_ids_data.into_boxed_slice()))
                .map_err(|e| InferenceError::Onnx(e.to_string()))?;

        let attention_mask = Tensor::from_array((
            [1usize, MAX_SEQ_LENGTH],
            attention_mask_data.into_boxed_slice(),
        ))
        .map_err(|e| InferenceError::Onnx(e.to_string()))?;

        // Lock session for inference
        let mut session = self
            .text_session
            .lock()
            .map_err(|e| InferenceError::Onnx(format!("Session lock error: {e}")))?;

        // Get output name before running (to avoid borrow conflicts)
        let output_name = session.outputs[0].name.clone();

        // Get actual input names from the model
        let input_names: Vec<String> = session.inputs.iter().map(|i| i.name.clone()).collect();
        debug!(?input_names, "Text model input names");

        // Build inputs based on what the model actually expects
        // Some CLAP models only need input_ids, others also need attention_mask
        let outputs = if input_names.len() == 1 {
            // Model only expects input_ids (e.g., some CLAP text encoders)
            session
                .run(ort::inputs![
                    input_names[0].as_str() => input_ids
                ])
                .map_err(|e| InferenceError::Onnx(e.to_string()))?
        } else if input_names.iter().any(|n| n.contains("attention")) {
            // Model expects both input_ids and attention_mask
            let id_input = input_names.iter().find(|n| n.contains("input_id") || n == &"input_ids").cloned()
                .unwrap_or_else(|| input_names[0].clone());
            let mask_input = input_names.iter().find(|n| n.contains("attention")).cloned()
                .unwrap_or_else(|| input_names[1].clone());

            session
                .run(ort::inputs![
                    id_input.as_str() => input_ids,
                    mask_input.as_str() => attention_mask
                ])
                .map_err(|e| InferenceError::Onnx(e.to_string()))?
        } else {
            // Fallback: try just input_ids with first input name
            session
                .run(ort::inputs![
                    input_names[0].as_str() => input_ids
                ])
                .map_err(|e| InferenceError::Onnx(e.to_string()))?
        };

        // Get first output (embedding)
        let output = outputs
            .get(output_name.as_str())
            .ok_or_else(|| InferenceError::Onnx(format!("Output '{}' not found", output_name)))?;

        // Extract tensor data - returns (shape, data_slice)
        let (shape, data) = output
            .try_extract_tensor::<f32>()
            .map_err(|e| InferenceError::Onnx(e.to_string()))?;

        debug!(?shape, data_len = data.len(), "Text model output");

        // Take first EMBEDDING_DIM values
        let result: Vec<f32> = data.iter().copied().take(EMBEDDING_DIM).collect();

        if result.len() < EMBEDDING_DIM {
            return Err(InferenceError::Onnx(format!(
                "Output too small: expected {}, got {}",
                EMBEDDING_DIM,
                result.len()
            )));
        }

        Ok(result)
    }

    /// Generate audio embedding from audio data
    pub fn audio_embedding(&self, audio: &AudioData) -> Result<Embedding, InferenceError> {
        let mut processor = self
            .audio_processor
            .lock()
            .map_err(|e| InferenceError::AudioProcessing(e.to_string()))?;

        // Preprocess audio to mel spectrogram windows
        let mel_features = processor.process(audio)?;

        if mel_features.is_empty() {
            return Err(InferenceError::AudioProcessing(
                "No mel spectrogram windows extracted".to_string(),
            ));
        }

        // Generate embedding for each window and average
        let mut embeddings: Vec<Vec<f32>> = Vec::new();

        for mel in &mel_features {
            let output = self.run_audio_inference(mel)?;
            embeddings.push(output);
        }

        // Average embeddings
        let mut averaged = vec![0.0f32; EMBEDDING_DIM];
        for emb in &embeddings {
            for (i, &v) in emb.iter().enumerate() {
                averaged[i] += v;
            }
        }
        let count = embeddings.len() as f32;
        for v in &mut averaged {
            *v /= count;
        }

        let mut embedding = Embedding::from_raw(averaged);
        embedding.normalize();

        Ok(embedding)
    }

    /// Run audio inference on pre-computed mel spectrogram features
    ///
    /// This is used by streaming to defer inference to session finalization.
    /// Returns raw embedding vector (not normalized).
    pub fn run_audio_inference_from_mel(
        &self,
        mel_features: &MelFeatures,
    ) -> Result<Vec<f32>, InferenceError> {
        self.run_audio_inference(mel_features)
    }

    fn run_audio_inference(&self, mel_features: &MelFeatures) -> Result<Vec<f32>, InferenceError> {
        // CLAP audio models expect input_features with shape [batch, channels, height, width]
        // where:
        // - batch = 1
        // - channels = 1 (mono mel spectrogram)
        // - height = spec_size (256 time frames, resized)
        // - width = n_mels (64 mel frequency bins)
        // Data is in row-major order [height, width] already

        let shape = [1usize, 1, mel_features.height, mel_features.width];
        let input = Tensor::from_array((shape, mel_features.data.clone().into_boxed_slice()))
            .map_err(|e| InferenceError::Onnx(e.to_string()))?;

        debug!(
            tensor_shape = ?shape,
            data_len = mel_features.data.len(),
            "Audio model input tensor"
        );

        // Lock session for inference
        let mut session = self
            .audio_session
            .lock()
            .map_err(|e| InferenceError::Onnx(format!("Session lock error: {e}")))?;

        // Get input and output names from the model
        let input_name = session.inputs[0].name.clone();
        let output_name = session.outputs[0].name.clone();

        let outputs = session
            .run(ort::inputs![
                input_name.as_str() => input
            ])
            .map_err(|e| InferenceError::Onnx(e.to_string()))?;

        // Get first output
        let output = outputs
            .get(output_name.as_str())
            .ok_or_else(|| InferenceError::Onnx(format!("Output '{}' not found", output_name)))?;

        let (shape, data) = output
            .try_extract_tensor::<f32>()
            .map_err(|e| InferenceError::Onnx(e.to_string()))?;

        debug!(?shape, data_len = data.len(), "Audio model output");

        let result: Vec<f32> = data.iter().copied().take(EMBEDDING_DIM).collect();

        if result.len() < EMBEDDING_DIM {
            return Err(InferenceError::Onnx(format!(
                "Audio output too small: expected {}, got {}",
                EMBEDDING_DIM,
                result.len()
            )));
        }

        Ok(result)
    }
}

/// Thread-safe wrapper for ClapModel
#[allow(dead_code)]
pub type SharedClapModel = Arc<ClapModel>;
