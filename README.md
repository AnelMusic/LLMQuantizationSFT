# Quantization and Supervised Fine-Tuning of LLMs

The fine-tuning of Large Language Models (LLMs) has seen remarkable progress in recent years. Hugging Face's introduction of the SFTTrainer class in the TRL library has greatly enhanced the developer experience, streamlining the fine-tuning process. In particular, when applying Quantized Low-Rank Adaptation (QLoRA), a systematic approach is used to drastically reduce the hardware requirements for these typically VRAM-intensive models.

In this notebook, you will learn how to load an (base) LLM from Hugging Face uncapable on answering questions, quantize it to 4-bit precision, and fine-tune it using QLoRA, a technique within the Parameter-Efficient Fine-Tuning (PEFT) framework. For this fine-tuning process, we’ll utilize a dataset from the Hugging Face Hub.

#### Base Model Output:
```
### INSTRUCTION:
You are an AI coding assistant specialized in generating Python code from user instructions.
Your task is to return only the code that directly fulfills the given instruction.</s>

### Input:
Design a Python code to Print the length of the string entered by user.</s>

### RESPONSE:

### Output:

### EXAMPLE:

### INPUT:

### OUTPUT:

...
```

#### Our Fine-Tuned Model Output:
```
### INSTRUCTION:
You are an AI coding assistant specialized in generating Python code from user instructions.
Your task is to return only the code that directly fulfills the given instruction.</s>

### Input:
Design a Python code to Print the length of the string entered by user.</s>

### RESPONSE:
string_length = len(input("Enter a string: "))
print("The length of the string is:", string_length)</s>
```


The training procedure involves a set of clearly defined and structured steps:


- ``Model Loading and Quantization``: Load the model onto GPU in 4-bit precision using the "bitsandbytes" library.

- ``LoRA Configuration``: Define the LoRA (Low-Rank Adaptation) configuration, which depends on the specific problem you're addressing.

- ``Training Hyperparameters``: Ultimately, as with every model, the success of fine-tuning depends on selecting appropriate training parameters.

- ``SFTTrainer Integration``: Finally, the defined training parameters are used with the SFTTrainer class to initiate the fine-tuning process.


### Quantization Configuration and Model Loading
The subsequent step in the fine-tuning process involves loading the specific model you intend to utilize. As previously indicated, our choice is the "llama2-7b," which constitutes the 7 billion parameter base model. However, it's worth noting that the Hugging Face model hub offers a diverse selection of models for various natural language processing tasks. Feel free to explore and experiment with different models available at https://huggingface.co/models to tailor your fine-tuning process to the specific requirements of your project. 

To ensure that our model undergoes quantization as intended, we'll use the BitsAndBytesConfig, serving as a wrapper class encapsulating various attributes and features available for manipulation when working with a loaded model through bitsandbytes. 

Presently, the BitsAndBytesConfig supports quantization methods such as LLM.int8(), FP4, and NF4. Should additional methods be incorporated into bitsandbytes in the future, corresponding arguments will be introduced to this class to accommodate these extensions.

Notably, bitsandbytes operates as a lightweight wrapper encompassing custom CUDA functions, particularly optimized for 8-bit operations, matrix multiplication (LLM.int8()), and quantization functions. For a deeper understanding of bitsandbytes and its functionalities, further information is available at: https://github.com/TimDettmers/bitsandbytes.

```python
# BitsAndBytes configuration for 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
```

- ```load_in_4bit=True```: This option ensures the model is quantized into 4-bit precision, significantly reducing the memory and computational requirements.

- ```bnb_4bit_use_double_quant=True```: Double quantization is applied, which quantizes the quantization constants themselves, providing further memory savings while maintaining model accuracy.

- ```bnb_4bit_quant_type="nf4"```: Specifies the quantization type. In this case, "nf4" stands for NormalFloat4, a quantization technique known for improved accuracy in lower precision formats.

- ```bnb_4bit_compute_dtype=torch.bfloat16```: The computation during inference and fine-tuning is performed using bfloat16, a data type that balances memory efficiency and numerical stability.
```python
model_name = "TinyPixel/Llama-2-7B-bf16-sharded"

# Load the model with safetensors
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    use_flash_attention_2=True,
    trust_remote_code=True,
    use_safetensors=True
)
```


- ```model_name```: This refers to the specific pre-trained LLM you want to load from Hugging Face’s model repository.

- ```quantization_config=bnb_config```: We apply the bnb_config to load the model with the 4-bit quantization settings defined earlier.

- ```device_map="auto"```: This automatically maps model layers to available hardware (e.g., CPU or GPU) for optimized performance and resource management.

- ```use_flash_attention_2=True```: FlashAttention is an optimized implementation of attention mechanisms that speeds up computation and reduces memory usage, especially useful for large models. The FlashAttention 2 further enhances this by supporting even faster and more efficient computations.

- ```trust_remote_code=True```: This allows loading model configurations that might include custom code directly from Hugging Face's model hub. It’s essential when working with models that contain unique implementations.

- ```use_safetensors=True```: By using safetensors, we load the model weights in a safe and efficient format that prevents malicious code injection, providing an added layer of security.

### LoRA Configuration:

We don't need to go into too much detail about LoRA because there's already a lot of information out there. If you’re interested feel to read the paper https://arxiv.org/abs/2106.09685. 

Here's the basic idea: the rank of a matrix tells us the number of linearly independent rows or columns it contains. This rank is important because it shows us the smallest amount of space needed to fit all the rows or columns.

Matrices suffering from rank deficiency, characterized by linear dependencies, inherently possess redundancy. In simple terms, this means we can express the same information using fewer dimensions (see https://arxiv.org/abs/2012.13255).

The basic idea behind **Low-Rank Adaptation (LoRA)** revolves around matrix **rank**, which refers to the number of linearly independent rows or columns in a matrix. This rank is critical because it determines the minimum number of dimensions needed to represent all the rows or columns without redundancy.

When a matrix suffers from **rank deficiency** (i.e., when there are linear dependencies among the rows or columns), it means some information is redundant. In practical terms, we can represent the same information using fewer dimensions. This is where LoRA comes into play, allowing us to exploit this redundancy to reduce the number of parameters we need to fine-tune (see [this paper](https://arxiv.org/abs/2012.13255) for more details).

In practice, instead of fine-tuning a large weight matrix directly, LoRA introduces two smaller matrices, often referred to as **A** and **B**. These matrices are much smaller in rank compared to the original weight matrix. During training, these two smaller matrices are trained, and when multiplied together, they approximate the original large matrix. The key idea here is that by choosing a low rank for matrices A and B, we can significantly reduce the number of parameters that need to be updated during fine-tuning. 

This reduction in parameters results in a more memory-efficient model that can be fine-tuned on hardware with limited resources, like consumer GPUs. However, there's a trade-off: reducing the number of "effective" parameters can potentially result in a **loss of information**. The LoRA hypothesis, however, suggests that this information loss is not a major concern because many parameters in the original weight matrix may not contribute significantly to the model's performance. In other words, a large portion of the original model’s parameters may be redundant, and LoRA can focus on updating only the most important components.

By using low-rank matrices, LoRA provides a parameter-efficient way to fine-tune large language models (LLMs) while still maintaining much of their performance, especially in tasks where a full fine-tuning might be overkill. Finally, we will prepare the model for training using the prepare_model_for_kbit_training() method. 

Here is what prepare_mode_for_kbit_training() does:

- It initiates the freezing of all model parameters by setting their gradient requirement to False, effectively preventing updates during the training process.

- For models that aren't quantized using the GPTQ method, it transforms parameters that are originally in formats such as 16-bit or bfloat16 into 32-bit floating-point format (fp32).

- If the model is initially loaded with lower bit precision (like 4-bit or 8-bit) or is quantized, and gradient checkpointing is enabled, it ensures that the inputs to the model necessitate gradients for training. This is achieved either by activating an existing function within the model or by registering a forward hook to the input embeddings to make sure their outputs require gradients.

- It conducts a compatibility check with the provided gradient checkpointing keyword arguments and provides a warning if the model doesn't support them.

- Ultimately, if all conditions align, the function enables gradient checkpointing with the appropriate parameters, thereby optimizing memory usage during training. This preparation proves especially valuable when training larger models or working with hardware that has limited memory resources.

First, to understand which target_moduls to include we can print the model:
```python
print(model)
```
```

LlamaForCausalLM(
  (model): LlamaModel(
    (embed_tokens): Embedding(32000, 4096, padding_idx=0)
    (layers): ModuleList(
      (0-31): 32 x LlamaDecoderLayer(
        (self_attn): LlamaAttention(
          (q_proj): Linear4bit(in_features=4096, out_features=4096, bias=False)
          (k_proj): Linear4bit(in_features=4096, out_features=4096, bias=False)
          (v_proj): Linear4bit(in_features=4096, out_features=4096, bias=False)
          (o_proj): Linear4bit(in_features=4096, out_features=4096, bias=False)
          (rotary_emb): LlamaRotaryEmbedding()
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear4bit(in_features=4096, out_features=11008, bias=False)
          (up_proj): Linear4bit(in_features=4096, out_features=11008, bias=False)
          (down_proj): Linear4bit(in_features=11008, out_features=4096, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): LlamaRMSNorm((4096,), eps=1e-05)
        (post_attention_layernorm): LlamaRMSNorm((4096,), eps=1e-05)
      )
    )
    (norm): LlamaRMSNorm((4096,), eps=1e-05)
    (rotary_emb): LlamaRotaryEmbedding()
  )
  (lm_head): Linear(in_features=4096, out_features=32000, bias=False)
)
```
In LoRA (Low-Rank Adaptation) fine-tuning, we usually target specific layers for parameter-efficient fine-tuning, focusing on attention mechanisms and linear projections.

Self-Attention Layers ```(self_attn)```: LoRA is often applied to the query ```(q_proj)```, key ```(k_proj)```, value ```(v_proj)```, and output ```(o_proj)``` projection matrices in attention mechanisms. These are the linear transformations in the attention mechanism that LoRA modifies by introducing low-rank decompositions. 

In our case, the following layers are target candidates:

- q_proj: 
- k_proj: 
- v_proj: 
- o_proj: 

These projection layers are part of the LlamaAttention module, where LoRA can be applied to efficiently adapt these parameters during fine-tuning.

Another common target for LoRA fine-tuning is the MLP ```(multi-layer perceptron)``` sub-layers that follow the attention mechanism. Specifically, the up projection and down projection matrices are often modified. In your model, the following linear layers within the MLP are candidates for fine-tuning with LoRA:

- ```gate_proj```
- ```up_proj```
- ```down_proj```

These projections control the transformations in the model's feed-forward layers, which are another effective place for LoRA's low-rank adaptation to reduce the number of trainable parameters while maintaining model performance.



```python
config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=[
                  "q_proj",
                  "up_proj",
                  "o_proj",
                  "k_proj",
                  "down_proj",
                  "gate_proj",
                  "v_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Gradient checkpointing is a technique used to trade off memory usage for 
# computation time during backpropagation
model.gradient_checkpointing_enable()

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, config)
```
### Training Hyperparameters

The following step is rather straightforward. Here, we simply configure the training settings. The values employed in this step are commonly used as standard starting points, but it's important to note that they may need adjustment for optimal performance based on your specific requirements.


```python

# Set training parameters
training_arguments = TrainingArguments(
    output_dir="./results",
    num_train_epochs= 3,
    per_device_eval_batch_size=5, # on RTX4090
    per_device_train_batch_size=5, # may need adjustment on your GPU
    gradient_accumulation_steps=2,
    gradient_checkpointing=True,
    save_strategy="no", 
    logging_strategy="steps",
    overwrite_output_dir=True,
    logging_steps=1,
    learning_rate=2e-4,
    weight_decay=0.001,
    bf16=True,  
    fp16=False, 
    tf32=False, 
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="constant",
    eval_strategy="steps",  
    eval_steps=1,
    disable_tqdm=False,
    seed=42,
    save_total_limit=None
)
```

Here’s an explanation of the key parameters in the `TrainingArguments` section, focusing on how different GPU architectures should set precision formats like `bf16`, `fp16`, and `tf32`:

### Key Parameters Explanation:

- ```output_dir="./results"```: This specifies the directory where the model's output (such as checkpoints and logs) will be saved.
  
- ```num_train_epochs=3```: The number of times the model will pass through the entire training dataset. Setting this to 3 means the model will train for 3 epochs.

- ```per_device_eval_batch_size=5` and ```per_device_train_batch_size=5``` 
  These set the batch size for training and evaluation on each GPU. In this example, a batch size of 5 is used on an ``RTX 4090`` GPU, but this may need adjustment depending on your GPU's available memory. Larger GPUs with more VRAM (e.g., ``A100`` or ``V100``) can handle larger batch sizes, while smaller GPUs (e.g., ``RTX 3080``) may need smaller ones.

- ```gradient_accumulation_steps=2```: This controls how many batches should be accumulated before performing a gradient update. Accumulating gradients over 2 steps effectively doubles the batch size without needing more memory, which is helpful when working with large models on GPUs with limited VRAM.

- ```gradient_checkpointing=True```: Enables checkpointing of gradients to save memory by offloading part of the model's memory consumption during training. This is useful when training large models like LLMs.

- ```save_strategy="no"```: Disables saving of model checkpoints. In certain cases, this is used to prevent unnecessary storage usage when checkpoints are not needed.

- ```logging_strategy="steps"```: Controls how often the training process logs information (such as loss or learning rate). Here, it logs at every step.

- ```overwrite_output_dir=True```: This allows the output directory to be overwritten if it already contains previous results. Useful when you are iterating on experiments.

- ```learning_rate=2e-4```: Specifies the initial learning rate for training. This is a critical hyperparameter that affects how fast or slow the model learns.

- ```weight_decay=0.001```: Applies regularization to the model weights to prevent overfitting by penalizing large weights.

##### Precision Settings for Different GPU Architectures:

- ```bf16=True```: 
   - This enables training in ``bfloat16`` precision, a format that is more memory efficient than `fp32` but more robust than `fp16` for certain operations. It's recommended for:
     - ``NVIDIA A100`` and ``H100``: These GPUs have native support for `bfloat16`, making this format ideal for efficient training on these architectures.
   - Example GPUs: ``A100, H100``

- ```fp16=False```: 
   - This disables training in ``half-precision (fp16)``, another format that reduces memory usage and accelerates training on compatible GPUs. However, since `bf16` is enabled, `fp16` is not necessary in this case.
   - If your GPU doesn't support `bf16`, consider setting `fp16=True` for faster and more memory-efficient training. This is typically supported on GPUs like:
     - ``RTX 3090``, ``RTX 3080``, ``V100``, and ``T4``
   - Example GPUs: ``V100, RTX 3090, RTX 3080``

- ```tf32=False```: 
   - ``TensorFloat32 (tf32)`` is a precision format that strikes a balance between accuracy and performance by using a mix of 19-bit mantissa and 8-bit exponent. You may want to enable this on ``Ampere-based GPUs`` (like ``A100``, ``RTX 3090``, ``RTX 3080``) when you need fast training but still want precision close to `fp32`. 
   - In this configuration, `tf32` is set to `False`, because `bf16` is already enabled and is preferred on GPUs that support both.
   - Example GPUs: ``A100, A10, RTX 3090, RTX 3080``

### Other Parameters:

- ```max_grad_norm=0.3```: This caps the gradient norm during backpropagation to prevent exploding gradients, a common issue in deep learning.

- ```warmup_ratio=0.03```: This defines the proportion of training steps during which the learning rate will be increased linearly from zero to the target learning rate. A small warm-up helps stabilize training early on.

- ```lr_scheduler_type="constant"```: The learning rate scheduler type. Here, a constant learning rate is used after the warm-up period.

- ```eval_strategy="steps"` and `eval_steps=1```: Specifies that evaluation should happen after every training step (useful for very short training runs or debugging).

- ```disable_tqdm=False```: Controls whether to disable the progress bar during training. Set to `False`, which means the progress bar is enabled.

- ```seed=42```: Sets the seed for random number generation to ensure reproducibility of results.

- ```save_total_limit=None```: No limit is set for the number of saved checkpoints, but since the saving strategy is disabled, this has no effect.

### SFTTrainer and Tokenizer Configuration:

Lastly, we can proceed with the setup of the Trainer. The Trainer class offers a comprehensive API for PyTorch training, suitable for most common scenarios. The SFTTrainer is an extension of the original transformers.Trainer class, designed to accommodate the direct initialization of the model for Parameter-Efficient Fine-Tuning (PEFT) by accepting the peft_config parameter.

You can make use of the formatting_func to structure your dataset samples in a way that fits your requirements. There are two options:  In case you’re using an already fine-tuned model you must stick with the existing prompt format. Alternatively, if you're working with the base model, you can define a prompt format that suits your specific needs. 

In or case the formatting_func is fairly simple. We add an instruction that should serve as a system prompt followed by the code and explanation blocks. Lastly, we can proceed with the setup of the Trainer. The Trainer class offers a comprehensive API for PyTorch training, suitable for most common scenarios. The SFTTrainer is an extension of the original transformers.Trainer class, designed to accommodate the direct initialization of the model for Parameter-Efficient Fine-Tuning (PEFT) by accepting the peft_config parameter.


```python
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
```

- ```tokenizer.pad_token = tokenizer.eos_token```: 
   - This line sets the ``pad token`` to be the same as the ``end-of-sequence (eos) token``. Typically, a ``pad token`` is used to fill empty positions in a batch of sequences, ensuring all sequences are the same length when fed into the model.
   - By setting the ``pad token`` to the ``eos token``, we ensure that when padding occurs, the padded positions are treated as if the sequence has ended. This is useful for causal language models like LLMs, which don’t need to differentiate between padding and actual end-of-sequence markers.

- ```tokenizer.padding_side = "right"```: 
   - This specifies that the padding will occur on the ``right side`` of the input sequence. Padding on the right is often preferred for autoregressive models (like LLaMA) since the model can focus on the left (the non-padded portion) and generate outputs without being distracted by the padded tokens.

#### Trainer Setup:

```python
trainer = SFTTrainer(
    model=model,
    train_dataset=debug_dataset_dict_1k["train"],
    eval_dataset=debug_dataset_dict_1k["test"],
    peft_config=config,
    formatting_func=format_instruction,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=True, # true
    max_seq_length=512  # 2048 reduced to reduce memory footprint
)
```

- ```train_dataset=debug_dataset_dict_1k["train"]``` and ```eval_dataset=debug_dataset_dict_1k["test"]```: 
   - These define the training and evaluation datasets. Here, we're using a debug dataset with 1,000 examples (`debug_dataset_dict_1k`) to quickly iterate and test the model during fine-tuning.

- ```peft_config=config```: 
   - This specifies the configuration for ``PEFT`` (Parameter-Efficient Fine-Tuning), which allows you to fine-tune the model efficiently by adjusting a smaller subset of parameters, such as through LoRA (Low-Rank Adaptation).

- ```formatting_func=format_instruction```: 
   - This is a custom formatting function that prepares each input into the correct format for instruction-based fine-tuning. It ensures that each sample from the dataset is processed in a way the model can understand.

- ```tokenizer=tokenizer```: 
   - The tokenizer you've defined earlier is passed to the trainer. It ensures that the input data is tokenized according to the tokenizer's configuration (e.g., padding on the right and using `eos_token` as `pad_token`).

- ```args=training_arguments```: 
   - The training arguments, previously defined in the `TrainingArguments`, control various aspects of the training process, such as batch size, learning rate, precision, etc.

- ```packing=True```: 
   - Setting `packing=True` means that multiple short sequences will be packed together into a single input to maximize the use of input tokens per batch. This helps improve computational efficiency by reducing unused space in the sequence.

- ```max_seq_length=512```: 
   - This sets the maximum sequence length to ``512 tokens`` for training. Reducing the maximum sequence length from larger values like ``1024`` or ``2048`` will reduce memory requirements and speed up training. 
   - ``Trade-off``: While reducing the sequence length helps with faster training and less memory usage, it may negatively affect model performance on long-context tasks. Texts longer than 512 tokens will be ``truncated``, which means the model will lose parts of the input context, potentially affecting the accuracy of the predictions. However, this truncation is specifically problematic for tasks requiring long-range context and may not always be an issue for shorter inputs.

- ```model.config.use_cache = False```: 
   - This disables the ``use of cached key/value states`` during training. Cached key/value states are typically used during inference to speed up generation in autoregressive models by storing attention activations. However, during fine-tuning, it’s generally better to disable caching to ensure proper gradient updates across all tokens in the sequence.


### Training
```python
trainer.train()
```
Training will take about 5 min on a RTX4090 using EarlyStopping. During training, you should observe a relatively gradual and consistent decrease in the training loss. It's essential to note that for fine-tuning intended for production use, you would typically incorporate a better split validation dataset as well as test set to ensure the model's performance is well-monitored and evaluated.

### Inference:
After training we can load the adapter and merge it with the base model. First, the model is saved under the name `"3_epoch_fine_tuned_laama"`. Then, `model.config.use_cache` is set to `True` to enable caching, which speeds up the inference process by storing intermediate attention activations. A prompt is generated using the test dataset, and a text generation pipeline is created using the fine-tuned model and tokenizer. Finally, the model generates text based on the prompt, and the generated text is printed.

```python

# Function to generate the prompt:
def generate_prompt(user_input):
    return f"""### INSTRUCTION:
You are an AI coding assistant specialized in generating Python code from user instructions.
Your task is to return only the code that directly fulfills the given instruction.</s>

### Input:
{user_input}</s>

### RESPONSE:
"""

new_model_name = "3_epoch_fine_tuned_laama"
trainer.model.save_pretrained(new_model_name)

model.config.use_cache = True

prompt = generate_prompt(debug_dataset_dict["test"][0]["instruction"])

# Run text generation pipeline with our next model
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=512)
result = pipe(f"{prompt}")
print(result[0]['generated_text'])
```

The result should look like:

```
### INSTRUCTION:
You are an AI coding assistant specialized in generating Python code from user instructions.
Your task is to return only the code that directly fulfills the given instruction.</s>

### Input:
Design a Python code to Print the length of the string entered by user.</s>

### RESPONSE:
string_length = len(input("Enter a string: "))
print("The length of the string is:", string_length)</s>
```

### Conclusion:

The process of fine-tuning a model on your own data is relatively straightforward. Nevertheless, the real challenge lies in evaluating these models and effectively monitoring them in production. Fortunately, the MLOps community is actively working on developing tools and best practices to streamline these procedures. As the field continues to evolve, we can expect greater support and resources to make the deployment and management of fine-tuned models more efficient and reliable.
