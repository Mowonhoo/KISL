# -*- coding: UTF-8 -*-
import pandas as pd
import numpy as np

def is_query_class(x,query, case_sensitive=False):
    # Format single search queries
    if type(query) == str:
        query = [query]
    # Remove case if necessary
    x_classname = str(x.__class__)
    if not case_sensitive:
        x_classname = x_classname.lower()
        query = map(lambda q:q.lower(),query)
    # Check if any of the tags in query are in the input class
    verdict = any(q in x_classname for q in query)
    return verdict

def is_symmetrical(X, tol=None, nans_ok=True):
    assert len(X.shape) == 2 , "`X` must be 2-dimensional"
    assert X.shape[0] == X.shape[1], "`X` must be square"
    
    X = X.copy()
    if nans_ok:
        X = X.fillna(0)
 
    if isinstance(X, pd.DataFrame):
        X = X.values
    np.fill_diagonal(X, 0)
    
    if tol is None:
        return np.all(np.tril(X) == np.triu(X).T)
    if tol:
        return (np.tril(X) - np.triu(X).T).ravel().min() < tol
    
# Topological overlap
def TOMsimilarity(adjacency, tol=1e-10):
    """
    Compute the topological overlap for a weighted adjacency matrix

    ====================================================
    Benchmark 5000 nodes (iris w/ 4996 noise variables):
    ====================================================
    TOM via rpy2 -> R -> WGCNA: 24 s ± 471 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    TOM via this function: 7.36 s ± 212 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

    =================
    Acknowledgements:
    =================
    Original source:
        * Peter Langfelder and Steve Horvath
        https://www.rdocumentation.org/packages/WGCNA/versions/1.67/topics/TOMsimilarity
        https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-9-559

    Implementation adapted from the following sources:
        * Credits to @scleronomic
        https://stackoverflow.com/questions/56574729/how-to-compute-the-topological-overlap-measure-tom-for-a-weighted-adjacency-ma/56670900#56670900
        * Credits to @benmaier
        https://github.com/benmaier/GTOM/issues/3
    """
    # Compute topological overlap
    def _compute_tom(A):
        # Prepare adjacency
        np.fill_diagonal(A, 0)
        # Prepare TOM
        A_tom = np.zeros_like(A)
        # Compute TOM
        L = np.matmul(A,A)
        ki = A.sum(axis=1)
        kj = A.sum(axis=0)
        MINK = np.array([ np.minimum(ki_,kj) for ki_ in ki ])
        A_tom = (L+A) / (MINK + 1 - A)
        np.fill_diagonal(A_tom,1)
        return A_tom

    # Check input type
    node_labels = None
    if not isinstance(adjacency, np.ndarray):
        if is_query_class(adjacency, "Symmetric"):
            adjacency = adjacency.to_dense()
        assert np.all(adjacency.index == adjacency.columns), "`adjacency` index and columns must have identical ordering"
        node_labels = adjacency.index

    # Check input type
    assert is_symmetrical(adjacency, tol=tol), "`adjacency` is not symmetric"
    assert np.all(adjacency >= 0), "`adjacency` weights must ≥ 0"

    # Compute TOM
    A_tom = _compute_tom(np.asarray(adjacency))

    # Unlabeled adjacency
    if node_labels is None:
        return A_tom

    # Labeled adjacency
    else:
        return pd.DataFrame(A_tom, index=node_labels, columns=node_labels)
    