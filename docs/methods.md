# Method summary

This is a short operational description of what the pipeline computes. For the
full motivation and validation, see the paper *(citation TBD)*.

## 1. Reference calls

A set of *reference calls* is sampled from the dataset (those whose species are
present in the tree). Each is analysed independently: we ask how far back in time
its call type can be traced.

## 2. Similarity → match probability

For a reference call, we compute the cosine similarity (a dot product, since
embeddings are L2-normalized) to every call in the dataset. For each species we
take the maximum similarity to that species' calls, and map it to a probability
that the species has the call type, using a fitted link function
`Pr(match | similarity)`:

- **histogram** — empirical match rate in similarity bins (with interpolation),
- **logit** — regularized logistic regression on similarity.

The reference call's own species is fixed to probability 1.

## 3. Ancestral-state reconstruction

The per-species probabilities are a trait over the tips of the dated tree. We
reconstruct the trait at internal (ancestral) nodes in two ways:

- **binary** — probabilities are thresholded to presence/absence at
  `leaf_threshold`, then reconstructed with `ape::ace` (discrete, equal-rates).
- **continuous** — raw probabilities are used as tip priors in a two-state
  Markov model reconstructed with `castor::asr_mk_model` (equal-rates). Tips
  without data get a flat 50/50 prior.

R is invoked via `Rscript`; the R script emits, per internal node, the
reconstructed likelihood that the call type is present, keyed by node label.

## 4. Age of a call

On the tree pruned to the species carrying the trait, the age of an internal
node is `max_leaf_depth − depth(node)` — time before the present in the tree's
branch-length units (Myr for a TimeTree chronogram). A node counts as "present"
when its reconstructed likelihood ≥ `ancestor_threshold`. The reported age of the
reference call is the **oldest** present internal node (or none, if no internal
node clears the threshold).
