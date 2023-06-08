# wilcoxauc
A fast Wilcoxon rank sum test and auROC analysis python package for single cell RNA-seq

## Usage
```
from wilcoxauc import wilcoxauc,top_markers

res = wilcoxauc(adata, group_name='leiden_0.5', layer='X')
res_top_markers = top_markers(res, n=10, auc_min=0.5, pval_max=0.01, padj_max=0.01)

```
