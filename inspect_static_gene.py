import joblib
import jax

params = joblib.load(
    r"checkpoints\bulk_rna_bert_tcga\params.joblib"
)

print("=" * 70)
print("PARAMETER TREE")
print("=" * 70)

flat = jax.tree_util.tree_leaves(params)

print("Number of parameter arrays:", len(flat))

print("\nPARAMETER MODULES:")

for module_name, module_params in params.items():

    print(f"\n{module_name}")

    if isinstance(module_params, dict):

        for param_name, value in module_params.items():

            if hasattr(value, "shape"):
                print(
                    f"  {param_name}: {value.shape}"
                )