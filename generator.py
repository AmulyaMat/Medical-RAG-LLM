"""
generator.py
------------
Craft prompt and call LLM to generate answers from retrieved context.
"""

from typing import Dict, List

import pandas as pd

import config
import context_builder
import llm_client


def generate_answer(question: str, rows: pd.DataFrame, patient_id: str = None) -> dict:
    """
    Generate an answer using the retrieved context.
    
    Args:
        question: User's question
        rows: DataFrame with retrieved chunks
        patient_id: Optional patient ID for context
    
    Returns:
        Dict with keys: answer (str), sources (list of lines), chunks (list of row dicts)
    """
    # Build context from retrieved rows
    context_blocks, sources_list = context_builder.build_context(rows)
    
    # Format sources as list
    sources = sources_list.split("\n") if sources_list else []
    
    # Convert rows to list of dicts for chunks
    chunks = rows.to_dict("records")
    
    # Craft user prompt
    user_prompt = f"Question: {question}\n"
    
    if patient_id:
        user_prompt += f"Patient ID: {patient_id}\n"
    
    user_prompt += f"\nContext:\n{context_blocks}\n\n"
    user_prompt += "Return:\n"
    user_prompt += "- A direct answer in 2–6 sentences.\n"
    user_prompt += "- Use bracketed numeric citations like [1], [2] that map to Sources.\n"
    user_prompt += "- If information is not present, say 'Not found in chart.'\n\n"
    user_prompt += f"Sources:\n{sources_list}"
    
    # Call LLM
    answer = llm_client.chat(config.SYSTEM_PROMPT, user_prompt)
    
    return {
        "answer": answer,
        "sources": sources,
        "chunks": chunks
    }
