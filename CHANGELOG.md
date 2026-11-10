# Changelog

## 0.3.0

### Anonymization

- Raise `AnonymizationError` when LLM PII anonymization fails or returns an invalid response, including batch processing. Callers relying on the previous original-text fallback must now handle this exception.
- Share one M2M100 model across translation stages and serialize calls to protect shared state.
- Load only active StyleRemix adapter constituents.
- Reconstruct authorship batches without repeatedly removing list prefixes.
- Validate optional Presidio parameters and retain entity counter state across mapping lifecycles.

### Evaluation and visualization

- Batch CoLA sentence scoring and disable gradient tracking during inference.
- Vectorize aligned SBERT and LUAR similarities.
- Normalize de-identification output once per document and retain initialized IMDb author metadata.
- Correct trade-off plot coordinates, handle empty radar model selections, and reuse model discovery in correlation calculations.
- Use the supported Matplotlib colormap API so trade-off plots work with current supported dependencies.
- Keep dependencies on compatible Transformers and Datasets major versions.

### Documentation and releases

- Clarify metric return values, CoLA loader order and empty-text behavior, de-identification recall direction, and original-baseline selection.
- Correct public configuration imports and editable installation instructions.
- Align package metadata, runtime version, documentation, and the PyPI badge at 0.3.0.
- Validate release versions and built distribution metadata before publishing to PyPI.
- Include LiteLLM and PEFT in the test extra, and repair the experiment integration-test dataset fixture.
