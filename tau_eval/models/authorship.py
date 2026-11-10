from threading import Lock

import torch
from nltk.tokenize import sent_tokenize

from transformers import pipeline

from .anonymizer import Anonymizer


class KeepItSimple(Anonymizer):
    def __init__(self, name="Keep-It-Simple"):
        self.name = name
        self.pipeline = pipeline(model="philippelaban/keep_it_simple", max_new_tokens=512)

    def anonymize(self, text: str) -> str:
        sentences = [t + "<|endoftext|>" for t in sent_tokenize(text)]
        anonymized_sentences = self.pipeline(sentences)

        anonymized_text = ""
        for i, sentence in enumerate(anonymized_sentences):
            anonymized_text += sentence[0]["generated_text"][len(sentences[i]) :]
        return anonymized_text

    def anonymize_batch(self, texts: list[str]) -> list[str]:
        sentences = []
        sentences_lenghts = []
        for text in texts:
            split = sent_tokenize(text)
            sentences_lenghts.append(len(split))
            sentences += [t + "<|endoftext|>" for t in split]

        anonymized_sentences = self.pipeline(sentences)
        anonymized_sentences = [
            sentence[0]["generated_text"][len(sentences[i]) :] for i, sentence in enumerate(anonymized_sentences)
        ]

        anonymized_texts = []
        for lenght in sentences_lenghts:
            text = " ".join([s for s in anonymized_sentences[:lenght]])
            del anonymized_sentences[:lenght]
            anonymized_texts.append(text)

        return anonymized_texts


class Paraphraser(Anonymizer):
    def __init__(self, name="PegasusParaphrase"):
        self.name = name
        self.pipeline = pipeline(
            "text2text-generation",
            model="alykassem/FLAN-T5-Paraphraser",
            max_new_tokens=512,
            torch_dtype=torch.float16,
        )

    def anonymize(self, text) -> str:
        sentences = sent_tokenize(text)
        anonymized_sentences = self.pipeline(sentences)
        anonymized_text = ""
        for i, sentence in enumerate(anonymized_sentences):
            anonymized_text += f" {sentence['generated_text']}"
        return anonymized_text

    def anonymize_batch(self, texts) -> list[str]:
        sentences = []
        sentences_lenghts = []
        for text in texts:
            split = sent_tokenize(text)
            sentences_lenghts.append(len(split))
            sentences += [t for t in split]

        anonymized_sentences = self.pipeline(sentences)
        anonymized_sentences = [sentence["generated_text"] for sentence in anonymized_sentences]

        anonymized_texts = []
        for lenght in sentences_lenghts:
            text = " ".join([s for s in anonymized_sentences[:lenght]])
            del anonymized_sentences[:lenght]
            anonymized_texts.append(text)

        return anonymized_texts


class M2M100MT(Anonymizer):
    """Round-trip translation with one model and a tokenizer per language stage.

    Calls to the single and batch APIs are serialized per instance to protect
    the shared model and mutable tokenizer state during generation.
    """

    def __init__(self, name="M2M100MT"):
        self.name = name
        self._translation_lock = Lock()
        checkpoint = "facebook/m2m100_418M"
        self.pipeline_en_de = pipeline("translation", checkpoint, src_lang="en", tgt_lang="de", max_new_tokens=512)
        # Passing the loaded model avoids loading the weights again. Explicit
        # tokenizer checkpoints give each stage its own language state.
        self.pipeline_de_fr = pipeline(
            "translation",
            self.pipeline_en_de.model,
            tokenizer=checkpoint,
            device=self.pipeline_en_de.device,
            src_lang="de",
            tgt_lang="fr",
            max_new_tokens=512,
        )
        self.pipeline_fr_en = pipeline(
            "translation",
            self.pipeline_en_de.model,
            tokenizer=checkpoint,
            device=self.pipeline_en_de.device,
            src_lang="fr",
            tgt_lang="en",
            max_new_tokens=512,
        )

    def _translate(self, sentences: list[str]) -> list[str]:
        with self._translation_lock:
            texts_de = [t["translation_text"] for t in self.pipeline_en_de(sentences)]
            texts_fr = [t["translation_text"] for t in self.pipeline_de_fr(texts_de)]
            return [t["translation_text"] for t in self.pipeline_fr_en(texts_fr)]

    def anonymize(self, text: str) -> str:
        sentences = sent_tokenize(text)
        texts_en = self._translate(sentences)
        anonymized_sentences = texts_en
        anonymized_text = ""
        for i, sentence in enumerate(anonymized_sentences):
            anonymized_text += f" {sentence}"
        return anonymized_text

    def anonymize_batch(self, texts) -> list[str]:
        sentences = []
        sentences_lenghts = []
        for text in texts:
            split = sent_tokenize(text)
            sentences_lenghts.append(len(split))
            sentences += [t for t in split]

        texts_en = self._translate(sentences)

        anonymized_sentences = texts_en
        anonymized_sentences = [sentence for i, sentence in enumerate(anonymized_sentences)]

        anonymized_texts = []
        for lenght in sentences_lenghts:
            text = " ".join([s for s in anonymized_sentences[:lenght]])
            del anonymized_sentences[:lenght]
            anonymized_texts.append(text)

        return anonymized_texts
