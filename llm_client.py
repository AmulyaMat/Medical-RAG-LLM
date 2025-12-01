"""
llm_client.py
-------------
Pluggable chat backend for different LLM providers.
"""

import re
from typing import Optional

import config


def chat(system_prompt: str, user_prompt: str) -> str:
    """
    Chat with the configured LLM backend.
    
    Args:
        system_prompt: System instruction prompt
        user_prompt: User question with context
    
    Returns:
        LLM response text
    """
    if config.LLM_BACKEND == "stub":
        return _stub_chat(system_prompt, user_prompt)
    elif config.LLM_BACKEND == "openai":
        return _openai_chat(system_prompt, user_prompt)
    elif config.LLM_BACKEND == "hf":
        return _hf_chat(system_prompt, user_prompt)
    else:
        raise ValueError(f"Unknown LLM backend: {config.LLM_BACKEND}")


def _stub_chat(system_prompt: str, user_prompt: str) -> str:
    """Stub implementation that returns first context block as answer."""
    # Extract context blocks from user prompt
    context_match = re.search(r"Context:\s*(.*?)(?=\n\nReturn:|$)", user_prompt, re.DOTALL)
    
    if context_match:
        context_text = context_match.group(1).strip()
        # Find first numbered block
        first_block_match = re.search(r"\[1\]\s*(.*?)(?=\n\n\[2\]|$)", context_text, re.DOTALL)
        if first_block_match:
            answer = first_block_match.group(1).strip()
            return f"{answer}\n\n[Citation: see Sources [1]]"
    
    return "Not found in chart."


def _openai_chat(system_prompt: str, user_prompt: str) -> str:
    """OpenAI API implementation."""
    try:
        import openai
        
        response = openai.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=600
        )
        return response.choices[0].message.content.strip()
    except ImportError:
        raise ImportError("OpenAI package not installed. Run: pip install openai")
    except Exception as e:
        return f"Error calling OpenAI API: {str(e)}"


def _hf_chat(system_prompt: str, user_prompt: str) -> str:
    """Hugging Face transformers implementation."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        # Load model and tokenizer
        model_name = config.HF_MODEL
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        # Format prompt
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        
        # Tokenize and generate
        inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=2048)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=600,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the new generated text
        response = response[len(full_prompt):].strip()
        
        return response if response else "Not found in chart."
        
    except ImportError:
        raise ImportError("Transformers package not installed. Run: pip install transformers torch")
    except Exception as e:
        return f"Error calling Hugging Face model: {str(e)}"
