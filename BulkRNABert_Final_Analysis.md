# BulkRNABert Representation Analysis and Validation

## Scope

This document records the BulkRNABert analysis completed against the
cloned `multiomics-open-research` repository and the pretrained
`bulk_rna_bert_tcga` checkpoint. It focuses on the three representation
levels requested for comparison with other transcriptomic foundation
models:

1.  **Static gene embedding**
2.  **Contextual gene embedding**
3.  **Sample embedding**

The dimensions, extraction points, preprocessing, and tensor shapes
below were validated directly from the repository implementation and by
running inference on the repository's bundled TCGA example. Runtime
values are from the completed local CPU benchmark.

------------------------------------------------------------------------

## 1. Model and checkpoint used

**Model:** BulkRNABert\
**Checkpoint:** `bulk_rna_bert_tcga`\
**Repository checkpoint files:**

-   `checkpoints/bulk_rna_bert_tcga/config.json`
-   `checkpoints/bulk_rna_bert_tcga/params.joblib`

The checkpoint configuration is:

  Parameter                                                                  Value
  ---------------------------------- ---------------------------------------------
  Number of genes                                                           19,062
  Expression bins                                                               64
  Transformer embedding dimension                                              256
  Initial gene embedding dimension                                             200
  Project gene embedding                                                      True
  Use gene embedding                                                          True
  Attention heads                                                                8
  Attention key size                                                            32
  FFN embedding dimension                                                      512
  Transformer layers                                                             4
  Memory-efficient attention                                                 False
  Gradient checkpointing                                                      True
  Log normalization                                                           True
  Max normalization                                                           True
  Normalization factor                                           5.547176906585117
  Gene2Vec weights path                `data/gene2vec_weights_common_gene_ids.npy`

The checkpoint parameter tree confirmed:

-   expression embedding: `(64, 256)`
-   static gene embedding: `(19,062, 200)`
-   gene projection weights: `(200, 256)`
-   final language-model head weights: `(256, 64)`

------------------------------------------------------------------------

## 2. Input data

The repository contains a ready-to-use TCGA example:

`data/bulkrnabert/tcga_sample.csv`

### Input shape

**4 samples × 19,065 columns**

The columns consist of:

-   `identifier`
-   `survival_time`
-   `event`
-   19,062 gene-expression features

The four bundled samples are:

  Sample ID            Survival time   Event
  ------------------ --------------- -------
  TCGA-06-2559-01A               150       1
  TCGA-06-0187-01A               828       1
  TCGA-27-2521-01A               510       1
  TCGA-06-0744-01A              1426       1

The model input itself is the **19,062-gene expression profile**.
`survival_time` and `event` are downstream labels and are not passed
into BulkRNABert during representation extraction.

Conceptually, the source table is:

``` text
identifier          survival_time  event  ENSG...  ENSG...  ... 19,062 genes
TCGA-06-2559-01A    150            1      value    value
TCGA-06-0187-01A    828            1      value    value
...
```

The fixed gene ordering is supplied by:

`data/bulkrnabert/common_gene_id.txt`

------------------------------------------------------------------------

## 3. Input preprocessing

For each sample:

``` text
19,062 expression values
        ↓
log10(1 + expression)
        ↓
BulkRNABert BinnedOmicTokenizer
        ↓
19,062 expression tokens
        ↓
BulkRNABert
```

The validation run produced:

-   raw expression shape: `(1, 19062)`
-   log-transformed expression shape: `(1, 19062)`
-   token shape: `(1, 19062)`
-   observed token range: `0–64`

The tokenizer is instantiated with `prepend_cls_token=False`.
BulkRNABert therefore does not depend on a CLS token to construct the
sample representation used by the downstream head.

------------------------------------------------------------------------

# 4. Three BulkRNABert representations

## 4.1 Static gene embedding

### Definition

BulkRNABert contains a learned gene embedding table:

``` python
self._gene_embedding_layer = hk.Embed(
    self._config.n_genes,
    self._config.init_gene_embed_dim,
    name="gene_embedding",
)
```

The pretrained checkpoint contains:

``` text
bulk_bert/~/gene_embedding
embeddings: (19062, 200)
```

### Shape

**19,062 genes × 200 dimensions**

Each gene has one learned vector. This representation is
**sample-independent**.

Conceptually:

``` text
Gene                 Static embedding
ENSG...              [d1, d2, ..., d200]
ENSG...              [d1, d2, ..., d200]
...
```

### Projection used by the model

The checkpoint has `project_gene_embedding=True`. A learned linear layer
projects the static embedding:

``` text
19,062 × 200
      ↓
Linear 200 → 256
      ↓
19,062 × 256
```

The parameter tree confirmed:

``` text
bulk_bert/~/linear
w: (200, 256)
b: (256,)
```

The projected 256-dimensional gene representation is added to the
expression-token embedding before transformer contextualization.

### Interpretation

This representation encodes **gene identity**. A given gene has the same
static embedding regardless of the patient/sample being processed.

------------------------------------------------------------------------

## 4.2 Contextual gene embedding

### Construction

For each sample, expression values are tokenized. BulkRNABert first
obtains an expression-bin embedding and combines it with the projected
gene embedding:

``` text
Expression-bin embedding
        +
Projected static gene embedding
        ↓
Transformer layers 1–4
        ↓
Final contextual gene representation
```

The model was configured during extraction with:

``` python
embeddings_layers_to_save=(4,)
```

The final transformer output is therefore returned as:

``` text
embeddings_4
```

### Verified output

The inference run returned:

``` text
embeddings_4: (1, 19062, 256)
logits:       (1, 19062, 64)
```

### Shape

For `N` samples:

**N × 19,062 genes × 256 dimensions**

For the single-sample benchmark:

**1 × 19,062 × 256**

### Interpretation

This representation is **sample-dependent**.

The same gene can have different contextual vectors in two patients
because its final representation is produced after transformer
processing of the complete sample expression-token sequence.

Conceptually:

``` text
Gene X in patient A → 256-D contextual vector A
Gene X in patient B → 256-D contextual vector B
```

This is the representation to use when the analysis requires
**gene-level information conditioned on the transcriptomic state of a
specific sample**.

------------------------------------------------------------------------

## 4.3 Sample embedding

The downstream BulkRNABert implementation receives embeddings with
shape:

``` text
(batch_size, seq_length, embedding_dimension)
```

It then transposes them and averages over the sequence/gene dimension.
Equivalently:

``` python
sample_embedding = mean(contextual_gene_embeddings, axis=genes)
```

### Shape

For `N` samples:

**N × 256**

For the validation sample:

**1 × 256**

### Verified example

For `TCGA-06-2559-01A`, the first dimensions of the resulting sample
embedding were:

``` text
0.53108424
0.07595378
0.78090817
0.16862464
0.36279327
0.05131939
-0.6253011
-0.28630468
0.01433209
-0.10094165
...
```

### Interpretation

The sample embedding compresses the final contextual representations of
all 19,062 genes into **one 256-dimensional representation of the
complete bulk RNA-seq sample**.

This is the natural representation for sample-level downstream tasks
such as survival prediction or classification.

------------------------------------------------------------------------

# 5. Representation flow

``` text
Bulk RNA-seq sample
19,062 gene-expression values
            │
            ▼
     log10(1 + expression)
            │
            ▼
       Tokenization
19,062 expression tokens
            │
            ├─────────────────────────────┐
            │                             │
            ▼                             ▼
Expression-bin embedding        Static gene embedding
      19,062 × 256                 19,062 × 200
                                          │
                                          ▼
                                   Linear projection
                                      200 → 256
                                          │
            └────────────── + ────────────┘
                           │
                           ▼
                Four transformer layers
                           │
                           ▼
               Contextual gene embedding
                  19,062 × 256/sample
                           │
                           ▼
                 Mean across genes
                           │
                           ▼
                    Sample embedding
                         256-D
```

------------------------------------------------------------------------

# 6. Representation comparison

  ------------------------------------------------------------------------------------------------------------------------
  Representation   Source                  Dimension Complete shape       Sample-dependent?   Primary interpretation
  ---------------- ------------------ -------------- -------------------- ------------------- ----------------------------
  Static gene      Checkpoint                    200 `19,062 × 200`       No                  Gene identity
                   `gene_embedding`                                                           

  Projected static Learned linear                256 `19,062 × 256`       No                  Gene identity in transformer
  gene             projection                                                                 input space

  Contextual gene  Final transformer             256 `N × 19,062 × 256`   Yes                 Gene state in sample context
                   layer                                                                      
                   `embeddings_4`                                                             

  Sample           Mean pooling of               256 `N × 256`            Yes                 Whole-transcriptome/sample
                   contextual genes                                                           representation
  ------------------------------------------------------------------------------------------------------------------------

The three primary representation levels for cross-model comparison are
**static gene (200-D), contextual gene (256-D), and sample (256-D)**.
The projected static representation is retained as an architectural
intermediate.

------------------------------------------------------------------------

# 7. Benchmark

The completed benchmark was run with JAX on **CPU**.

  ----------------------------------------------------------------------------------------------
  Tool          Embedding       Dimension Output       Method                     Time Peak GPU
                type                                                                   VRAM
  ------------- ------------ ------------ ------------ ------------------ ------------ ---------
  BulkRNABert   Static gene           200 19,062 gene  Checkpoint           0.002956 s N/A (CPU)
                                          embeddings   `gene_embedding`                
                                                       lookup                          

  BulkRNABert   Contextual            256 1 × 19,062   Final transformer   23.144672 s N/A (CPU)
                gene                      contextual   layer                           
                                          gene         (`embeddings_4`)                
                                          embeddings                                   

  BulkRNABert   Sample                256 1 sample     Mean pooling         0.029078 s N/A (CPU)
                                          embedding    across contextual               
                                                       gene embeddings                 
  ----------------------------------------------------------------------------------------------

### Timing interpretation

The three times measure different operations:

-   **Static gene:** retrieval of the already learned embedding matrix
    from the checkpoint.
-   **Contextual gene:** steady-state BulkRNABert inference for one
    sample after JIT warm-up.
-   **Sample:** mean pooling after the contextual representation already
    exists.

The contextual time therefore represents the computationally meaningful
foundation-model inference cost in this benchmark.

### GPU interpretation

The validated environment used CPU JAX, so GPU VRAM is correctly
reported as:

`N/A (CPU)`

These timings should be retained as **CPU benchmark results**. A
CUDA-enabled JAX run is required for a GPU runtime/VRAM comparison.

------------------------------------------------------------------------

# 8. Model output head

In addition to `embeddings_4`, the model returned:

``` text
logits: (1, 19062, 64)
```

The 64-wide final axis corresponds to the expression-bin prediction
head. This output is separate from the three representations used for
the representation comparison.

------------------------------------------------------------------------

# 9. Files generated by the representation notebook

The reproducible notebook is:

`BulkRNABert_3_representations_with_benchmark.ipynb`

It generates:

``` text
BulkRNABert_representations/
│
├── benchmark_summary.csv
│
├── snapshots/
│   ├── 00_tcga_expression_survival_snapshot.csv
│   ├── 01_static_gene_embedding_snapshot.csv
│   ├── 02_contextual_gene_embedding_snapshot.csv
│   └── 03_sample_embedding_snapshot.csv
│
└── full_outputs/
    ├── static_gene_embeddings.npy
    ├── projected_static_gene_embeddings.npy
    ├── contextual_gene_embeddings.npy
    └── sample_embeddings.npy
```

### Snapshot roles

**`00_tcga_expression_survival_snapshot.csv`**\
Shows the original model source data: sample identifier, survival
labels, and representative expression values.

**`01_static_gene_embedding_snapshot.csv`**\
Shows representative genes and dimensions from the 200-D static gene
representation.

**`02_contextual_gene_embedding_snapshot.csv`**\
Shows the same gene positions across multiple TCGA samples, allowing
direct inspection of sample-dependent contextualization.

**`03_sample_embedding_snapshot.csv`**\
Shows the 256-D sample representation together with sample ID, survival
time, and event label.

### Full outputs

The `.npy` files preserve the complete numerical representations without
expanding the large contextual tensor into an inefficient CSV.

------------------------------------------------------------------------

# 10. Downstream interpretation

The bundled TCGA data provide a direct example of how the representation
can connect to a downstream endpoint:

``` text
19,062-gene expression profile
        ↓
BulkRNABert
        ↓
256-D sample embedding
        ↓
downstream survival model
        ↓
survival/risk prediction

Target information:
survival_time + event
```

The sample representation is therefore the relevant representation for
patient/sample-level prediction.

The contextual representation supports analyses where the question
concerns **how individual genes are represented within a particular
transcriptomic context**.

The static representation supports analyses of **sample-independent gene
identity or relationships between learned gene vectors**.

------------------------------------------------------------------------

# 11. Final validated specification

## Static gene representation

-   **Input/source:** pretrained checkpoint parameter
-   **Genes:** 19,062
-   **Dimension:** 200
-   **Shape:** `19,062 × 200`
-   **Sample-specific:** No
-   **Projection:** 200 → 256 before transformer input
-   **Use:** gene-level sample-independent representation

## Contextual gene representation

-   **Input:** tokenized bulk expression profile
-   **Source:** fourth/final transformer layer
-   **Dimension:** 256
-   **Shape:** `N × 19,062 × 256`
-   **Sample-specific:** Yes
-   **Validated single-sample shape:** `1 × 19,062 × 256`
-   **Use:** sample-conditioned gene representation

## Sample representation

-   **Input:** final contextual gene embeddings
-   **Method:** mean pooling across all 19,062 gene positions
-   **Dimension:** 256
-   **Shape:** `N × 256`
-   **Sample-specific:** Yes
-   **Validated single-sample shape:** `1 × 256`
-   **Use:** sample-level downstream prediction

------------------------------------------------------------------------

## Reproducibility note

The validated local environment used:

``` text
pandas  2.2.0
jax     0.4.19
haiku   0.0.10
numpy   1.26.0
```

The PyArrow message emitted by pandas during execution was a deprecation
warning and did not affect the BulkRNABert analysis.
