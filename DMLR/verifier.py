import json
import os
from typing import Optional

from openai import OpenAI
from pydantic import BaseModel
import logging


_client: Optional[OpenAI] = None
_model_name: Optional[str] = None


def _load_dotenv_into_environ() -> None:
    cwd = os.path.dirname(os.path.abspath(__file__))
    # Walk up to repo root and load the first .env found
    current = cwd
    visited = set()
    while True:
        if current in visited:
            break
        visited.add(current)
        env_path = os.path.join(current, ".env")
        if os.path.exists(env_path):
            try:
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#") or "=" not in line:
                            continue
                        key, value = line.split("=", 1)
                        key = key.strip()
                        value = value.strip()
                        if key and value and key not in os.environ:
                            os.environ[key] = value
            except Exception:
                pass
            break
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent


def _get_client() -> OpenAI:
    global _client, _model_name
    if _client is not None:
        return _client

    _load_dotenv_into_environ()
    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_API_BASE_URL")
    _model_name = os.environ.get("MODEL_TYPE", "gpt-4o-2024-08-06")

    kwargs = {}
    if api_key:
        kwargs["api_key"] = api_key
    if base_url:
        kwargs["base_url"] = base_url

    _client = OpenAI(**kwargs)
    return _client


def verify_solution_equivalence(solution: str, ground_truth: str) -> bool:
    """
    Return True if the provided solution is equivalent to ground truth, judged by an LLM verifier.
    The API credentials and model are read from .env (OPENAI_API_KEY, OPENAI_API_BASE_URL, MODEL_TYPE).
    """
    client = _get_client()
    model = _model_name or "gpt-4o-2024-08-06"

    class EquivalenceResult(BaseModel):
        equivalent: bool

    logging.info(f"\x1b[32mSolution: {solution[:10]}\x1b[0m")
    logging.info(f"\x1b[32mGround truth: {ground_truth}\x1b[0m")

    try:
        if solution == "":
            return False
        if ground_truth == "":
            return False
        else:
            resp = client.chat.completions.parse(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            f"Compare the following two answers and decide if they express the same final result. Return True if they are the same, False otherwise."
                            f"Candidate answer: {solution}\n\n"
                            f"Ground truth: {ground_truth}\n\n"
                        ),
                    },
                ],
                response_format=EquivalenceResult,
                temperature=0,
            )
            parsed: EquivalenceResult = resp.choices[0].message.parsed
            logging.info(f"\x1b[32mParsed response: {bool(parsed.equivalent)}\x1b[0m")
            return bool(parsed.equivalent)
    except Exception as e:
        logging.error(f"Error verifying solution equivalence: {e}")
        return False
