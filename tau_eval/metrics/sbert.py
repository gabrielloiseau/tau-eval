import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import pairwise_cos_sim


def load_sbert(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "cuda"
) -> SentenceTransformer:
    """
    Loads the sentence similarity model.

    Args:
        model_name: SentenceTransformers model to load.
        device: The device to load the model onto ("cuda" or "cpu").

    Returns:
        The loaded SentenceTransformer model.
    """
    return SentenceTransformer(model_name, device=device)


def compute_sbert(
    input_texts: str | list[str],
    output_texts: str | list[str],
    sim_model: SentenceTransformer,
) -> dict[str, list[float]]:
    """
    Computes the cosine similarity between the embeddings of original and rewritten texts.

    Args:
        input_texts: A string or a list of original texts.
        output_texts: A string or a list of rewritten texts aligned with
            input_texts. Each corresponding pair is scored in input order.
        sim_model: The loaded SentenceTransformer model.

    Returns:
        A dictionary with key "sbert" containing a list of float cosine
        similarities, one per aligned input/output pair in input order.
        Single-string inputs produce a one-element score list.
    """
    if not isinstance(input_texts, list):
        input_texts = [input_texts]
    if not isinstance(output_texts, list):
        output_texts = [output_texts]
    assert len(input_texts) == len(output_texts), "inputs are different lengths"

    if not input_texts:
        return {"sbert": []}
    embedding_orig: torch.Tensor = sim_model.encode(input_texts, convert_to_tensor=True, show_progress_bar=False)
    embedding_rew: torch.Tensor = sim_model.encode(output_texts, convert_to_tensor=True, show_progress_bar=False)

    outputs = pairwise_cos_sim(embedding_orig, embedding_rew).cpu().tolist()

    return {"sbert": outputs}
