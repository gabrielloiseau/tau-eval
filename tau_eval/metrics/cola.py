import numpy as np
import torch
from nltk.tokenize import sent_tokenize

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)


def load_cola(
    model_name: str = "textattack/roberta-base-CoLA", device: str = "cuda"
) -> tuple[PreTrainedTokenizer, PreTrainedModel]:
    """
    Loads the CoLA (Corpus of Linguistic Acceptability) model and tokenizer.

    Args:
        model_name: HuggingFace model to load
        device: The device to load the model onto ("cuda" or "cpu").

    Returns:
        The tuple (tokenizer, model), in that order. Unpack it as
        cola_tokenizer, cola_model = load_cola(), then pass the tokenizer before
        the model to compute_cola(output_texts, cola_tokenizer, cola_model).
    """
    cola_tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_name, map_location=device)
    cola_model: PreTrainedModel = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    return (cola_tokenizer, cola_model)


def cola_score(
    text: str,
    cola_tokenizer: PreTrainedTokenizer,
    cola_model: PreTrainedModel,
    device: str = "cuda",
) -> float:
    """
    Calculates the CoLA score for a single piece of text.

    Args:
        text: The text to score.
        cola_tokenizer: The tokenizer for the CoLA model.
        cola_model: The CoLA model.
        device: The device to run the model on ("cuda" or "cpu").

    Returns:
        The CoLA score (a float between 0 and 1).
    """
    tokenize_input = cola_tokenizer.tokenize(text)
    tensor_input = torch.tensor([cola_tokenizer.convert_tokens_to_ids(tokenize_input)]).to(device)
    output = cola_model(tensor_input)
    return output.logits.softmax(-1)[0][1].item()


def compute_cola(
    output_texts: str | list[str],
    cola_tokenizer: PreTrainedTokenizer,
    cola_model: PreTrainedModel,
    device: str = "cuda",
    batch_size: int = 32,
) -> dict[str, list[float]]:
    """
    Computes CoLA scores for a list of input texts.

    Args:
        output_texts: A list of text strings.
        cola_tokenizer: The tokenizer for the CoLA model.
        cola_model: The CoLA model.
        device: The device to run the model on ("cuda" or "cpu").
        batch_size: Maximum number of sentences per classifier forward pass.

    Returns:
        A dictionary with key "cola" containing the sentence-mean score for
        each nonempty text, in relative input order. Exact empty strings ("")
        are omitted without a placeholder, so the list can be shorter than
        output_texts. The experiment evaluator averages only returned scores;
        omitted empty strings do not contribute to that average.

        For example, for ["First text.", "", "Third text."], if the first and
        third texts score 0.8 and 0.6, the result is {"cola": [0.8, 0.6]}.
        There is no entry between those scores for the exact empty string.
    """
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if not isinstance(output_texts, list):
        output_texts = [output_texts]

    sentences = []
    sentence_counts = []
    for text in output_texts:
        if text == "":
            continue
        text_sentences = sent_tokenize(text)
        sentence_counts.append(len(text_sentences))
        sentences.extend(text_sentences)

    scores = []
    for offset in range(0, len(sentences), batch_size):
        # Keep the original tokenization: do not add special tokens or truncate.
        token_ids = [
            torch.tensor(cola_tokenizer.convert_tokens_to_ids(cola_tokenizer.tokenize(sentence)), dtype=torch.long)
            for sentence in sentences[offset : offset + batch_size]
        ]
        # Right padding preserves unpadded token positions, irrespective of the
        # tokenizer's default padding side. Masks exclude only added padding.
        padding_id = cola_tokenizer.pad_token_id if cola_tokenizer.pad_token_id is not None else 0
        input_ids = torch.nn.utils.rnn.pad_sequence(token_ids, batch_first=True, padding_value=padding_id)
        lengths = torch.tensor([len(ids) for ids in token_ids])
        attention_mask = torch.arange(input_ids.shape[1])[None, :] < lengths[:, None]
        with torch.no_grad():
            output = cola_model(input_ids.to(device), attention_mask=attention_mask.to(device))
            scores.extend(output.logits.softmax(-1)[:, 1].cpu().tolist())

    cola_ls: list[float] = []
    offset = 0
    for count in sentence_counts:
        cola_ls.append(np.mean(scores[offset : offset + count]))
        offset += count

    return {"cola": cola_ls}
