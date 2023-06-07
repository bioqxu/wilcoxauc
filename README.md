# wilcoxauc
A fast Wilcoxon rank sum test and auROC analysis python package for single cell RNA-seq

## Usage
```
import wilcoxauc

res = wilcoxauc(adata, group_name, layer='X')
res_top_markers = top_markers(res)

```
