import numpy as np
import pandas as pd

import jax
from jax import numpy as jnp
from sklearn import preprocessing

import scanpy as sc

def get_expr(adata, layer=None):
    """Get expression matrix from adata object"""

    if layer is not None:
        x = adata.layers[layer]
    else:
        x = adata.raw.X
        
    if hasattr(x, "todense"):
        expr = jnp.asarray(x.todense())
    else:
        expr = jnp.asarray(x)

    return expr

@jax.jit
def jit_auroc(x, groups):

    # sort scores and corresponding truth values
    desc_score_indices = jnp.argsort(x)[::-1]
    x = x[desc_score_indices]
    groups = groups[desc_score_indices]

    # x typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = jnp.array(jnp.diff(x) != 0, dtype=jnp.int32)
    threshold_mask = jnp.r_[distinct_value_indices, 1]

    # accumulate the true positives with decreasing threshold
    tps_ = jnp.cumsum(groups)
    fps_ = 1 + jnp.arange(groups.size) - tps_

    # mask out the values that are not distinct
    tps = jnp.sort(tps_ * threshold_mask)
    fps = jnp.sort(fps_ * threshold_mask)
    tps = jnp.r_[0, tps]
    fps = jnp.r_[0, fps]
    fpr = fps / fps[-1]
    tpr = tps / tps[-1]
    area = jnp.trapz(tpr, fpr)
    return area

vmap_auroc = jax.vmap(jit_auroc, in_axes=[1, None])

def expr_auroc_over_groups(expr, groups):
    """Computes AUROC for each group separately."""
    auroc = np.zeros((groups.max() + 1, expr.shape[1]))

    for group in range(groups.max() + 1):
        auroc[group, :] = np.array(vmap_auroc(expr, groups == group))

    return auroc


def wilcoxauc(adata, group_name, layer=None):
    expr = get_expr(adata, layer=layer)

    # Turn string groups into integers
    le = preprocessing.LabelEncoder()
    le.fit(adata.obs[group_name])

    groups = jnp.array(le.transform(adata.obs[group_name]))

    auroc = expr_auroc_over_groups(expr, groups)

    if layer is not None:
        features = adata.var.index
        sc.tl.rank_genes_groups(adata, group_name, layer=layer, use_raw=False,
                        method='wilcoxon', key_added = "wilcoxon")
    else:
        features = adata.raw.index
        sc.tl.rank_genes_groups(adata, group_name, 
                                method='wilcoxon', key_added = "wilcoxon")

    auroc_df = pd.DataFrame(auroc).T
    auroc_df.index = features
    auroc_df.columns = range(groups.max() + 1)

    res=pd.DataFrame()
    for c in range(groups.max() + 1):
        cstast = sc.get.rank_genes_groups_df(adata, group=str(c), key='wilcoxon')
        cauc = pd.DataFrame(auroc_df[c]).reset_index().rename(columns={'index':'names', c:'auc'})
        cres = pd.merge(cstast, cauc, on='names')
        cres['group']=str(c)
        res = pd.concat([res, cres])
        
    res = res.reset_index(drop=True)
    
    return res
         
def top_markers(res, n=10, auc_min=0, pval_max=1, padj_max=1):
    groups = res.group.unique()
    
    res = res[(res.auc>auc_min) & (res.pvals<pval_max) & (res.pvals_adj<padj_max)]

    res_ntop=pd.DataFrame()
    for c in groups:
        ntop_genes = res[res.group==c].sort_values('auc', ascending=False).head(n).names.tolist()

        if len(ntop_genes)<n:
            ntop_genes.extend((n-len(ntop_genes)) *[np.nan])

        res_ntop[c] = ntop_genes

    return res_ntop
