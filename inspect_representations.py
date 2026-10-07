import pandas as pd
import jax
import jax.numpy as jnp
import haiku as hk

from multiomics_open_research.bulk_rna_bert.pretrained import (
    get_bulkrnabert_pretrained_model,
)


# ============================================================
# 1. Load pretrained BulkRNABert-TCGA
# ============================================================

params, forward_fn, tokenizer, config = get_bulkrnabert_pretrained_model(
    model_name="bulk_rna_bert_tcga",

    # Save output of the final transformer layer.
    # BulkRNABert-TCGA has 4 transformer layers.
    embeddings_layers_to_save=(4,),
)

print("=" * 70)
print("MODEL CONFIG")
print("=" * 70)

print("Genes:", config.n_genes)
print("Expression bins:", config.n_expressions_bins)
print("Embedding dimension:", config.embed_dim)
print("Initial gene embedding dimension:", config.init_gene_embed_dim)
print("Transformer layers:", config.num_layers)
print("Attention heads:", config.num_attention_heads)
print("Project gene embedding:", config.project_gene_embedding)


# ============================================================
# 2. Load the repository TCGA example
# ============================================================

file = r"data\bulkrnabert\tcga_sample.csv"

df = pd.read_csv(file)

print("\n" + "=" * 70)
print("TCGA DATA")
print("=" * 70)

print("Dataset shape:", df.shape)

print("\nSamples:")
print(df["identifier"].tolist())

print("\nMetadata:")
print(
    df[
        ["identifier", "survival_time", "event"]
    ].to_string(index=False)
)


# ============================================================
# 3. Extract expression matrix
# ============================================================

# Repository TCGA file contains:
#
# identifier
# survival_time
# event
# 19062 expression columns
#
# Therefore the expression matrix begins at column 4.

expression = df.iloc[:, 3:].to_numpy(dtype="float32")

print("\nExpression matrix shape:")
print(expression.shape)

assert expression.shape[1] == config.n_genes, (
    f"Expected {config.n_genes} genes, "
    f"but found {expression.shape[1]}"
)


# ============================================================
# 4. Use ONE sample for initial validation
# ============================================================

sample_id = df.iloc[0]["identifier"]

x = expression[:1]

print("\n" + "=" * 70)
print("SELECTED SAMPLE")
print("=" * 70)

print("Sample:", sample_id)
print("Raw expression shape:", x.shape)


# ============================================================
# 5. Apply BulkRNABert preprocessing
# ============================================================

# Repository inference workflow uses log10(1 + expression)
# before tokenization.

x = jnp.asarray(x)

x_log = jnp.log10(1.0 + x)

print("Log-transformed expression shape:", x_log.shape)


# ============================================================
# 6. Tokenize expression
# ============================================================

tokens = tokenizer.batch_tokenize(x_log)

print("\n" + "=" * 70)
print("TOKENIZATION")
print("=" * 70)

print("Token shape:", tokens.shape)
print("Minimum token:", int(tokens.min()))
print("Maximum token:", int(tokens.max()))

print("\nFirst 20 tokens:")
print(tokens[0, :20])


# ============================================================
# 7. Transform the Haiku forward function
# ============================================================

# build_bulk_rna_bert_forward_fn() returns a raw Haiku
# function. hk.transform() creates the .apply() interface
# required to use the pretrained parameter tree.

transformed = hk.transform(forward_fn)

apply_fn = jax.jit(transformed.apply)


# ============================================================
# 8. Run BulkRNABert
# ============================================================

print("\n" + "=" * 70)
print("RUNNING BULKRNABERT")
print("=" * 70)

outputs = apply_fn(
    params,
    None,       # RNG is not required for this inference pass
    tokens,
)


# ============================================================
# 9. Inspect model outputs
# ============================================================

print("\nMODEL OUTPUT KEYS:")
print(list(outputs.keys()))

print("\nOUTPUT SHAPES:")

for key, value in outputs.items():

    if hasattr(value, "shape"):
        print(f"{key}: {value.shape}")


# ============================================================
# 10. Contextual gene representation
# ============================================================

# embeddings_4 is the output of the final (4th)
# transformer layer.

contextual_gene_embeddings = outputs["embeddings_4"]

print("\n" + "=" * 70)
print("CONTEXTUAL GENE EMBEDDINGS")
print("=" * 70)

print("Shape:")
print(contextual_gene_embeddings.shape)

print(
    "\nExpected shape: "
    "(1 sample, 19062 genes, 256 dimensions)"
)

print("\nFirst gene, first 10 dimensions:")

print(
    contextual_gene_embeddings[
        0,
        0,
        :10
    ]
)


# ============================================================
# 11. Sample representation
# ============================================================

# The repository downstream model mean-pools across the
# sequence/gene dimension.
#
# contextual shape:
# (batch, genes, embedding dimensions)
#
# mean across axis=1:
# (batch, embedding dimensions)

sample_embedding = jnp.mean(
    contextual_gene_embeddings,
    axis=1,
)

print("\n" + "=" * 70)
print("SAMPLE EMBEDDING")
print("=" * 70)

print("Shape:")
print(sample_embedding.shape)

print("\nExpected shape: (1, 256)")

print("\nFirst 20 dimensions:")

print(
    sample_embedding[
        0,
        :20
    ]
)


# ============================================================
# 12. Final verification
# ============================================================

assert contextual_gene_embeddings.shape == (
    1,
    config.n_genes,
    config.embed_dim,
)

assert sample_embedding.shape == (
    1,
    config.embed_dim,
)

print("\n" + "=" * 70)
print("VALIDATION COMPLETE")
print("=" * 70)

print(
    "Contextual gene representation:",
    contextual_gene_embeddings.shape,
)

print(
    "Sample representation:",
    sample_embedding.shape,
)

print("\nSample ID:", sample_id)