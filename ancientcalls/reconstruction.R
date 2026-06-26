#!/usr/bin/env Rscript
# Ancestral-state reconstruction backend.
#
# Usage: Rscript reconstruction.R <tree.nwk> <traits.json> <mode> <out.json>
#
#   tree.nwk     dated Newick with NAMED internal nodes
#   traits.json  {call_key: {species: value}}  (values are 0/1 for binary mode,
#                raw probabilities in [0,1] for continuous mode)
#   mode         "binary" or "continuous"
#   out.json     {call_key: {node_label: presence_likelihood}}
#
# All calls are processed in one R session to amortise startup. Node
# likelihoods are keyed by the tree's internal node labels (NOT positional
# indices) so the Python side can map them to node ages.

suppressPackageStartupMessages({
  library(ape)
  library(castor)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 4) {
  stop("Usage: Rscript reconstruction.R <tree.nwk> <traits.json> <mode> <out.json>")
}
tree_path <- args[1]
traits_path <- args[2]
mode <- args[3]
out_json <- args[4]

tree <- read.tree(tree_path)
tips <- tree$tip.label
node_labels <- tree$node.label
traits <- fromJSON(traits_path, simplifyVector = FALSE)

reconstruct_binary <- function(vec) {
  # vec: named numeric over a subset of tips, values in {0, 1}.
  x <- setNames(as.numeric(vec), names(vec))
  x <- x[names(x) %in% tips]
  if (length(x) < 2 || length(unique(x)) < 2) {
    return(NULL)
  }
  res <- tryCatch(
    ace(x, tree, type = "discrete", model = "ER"),
    error = function(e) NULL
  )
  if (is.null(res)) {
    return(NULL)
  }
  lik <- res$lik.anc
  cols <- colnames(lik)
  col <- if ("1" %in% cols) "1" else cols[min(length(cols), 2)]
  setNames(as.list(lik[, col]), rownames(lik))
}

reconstruct_continuous <- function(vec) {
  # vec: named numeric over a subset of tips, values in [0, 1].
  tip_priors <- matrix(0.5, nrow = length(tips), ncol = 2)
  rownames(tip_priors) <- tips
  for (nm in names(vec)) {
    if (nm %in% tips) {
      p1 <- max(0, min(1, as.numeric(vec[[nm]])))
      tip_priors[nm, ] <- c(1 - p1, p1)
    }
  }
  fit <- tryCatch(
    asr_mk_model(
      tree = tree, tip_states = NULL, tip_priors = tip_priors,
      Nstates = 2, rate_model = "ER", Nthreads = 1
    ),
    error = function(e) NULL
  )
  if (is.null(fit) || is.null(fit$ancestral_likelihoods)) {
    return(NULL)
  }
  lik <- fit$ancestral_likelihoods
  cols <- colnames(lik)
  col_idx <- if (!is.null(cols) && "2" %in% cols) which(cols == "2")[1] else min(2, ncol(lik))
  # castor returns no row names; ancestral nodes are in tree$node.label order.
  setNames(as.list(lik[, col_idx]), node_labels)
}

out <- list()
for (call_key in names(traits)) {
  vec <- unlist(traits[[call_key]])
  node_vals <- if (mode == "binary") reconstruct_binary(vec) else reconstruct_continuous(vec)
  out[[call_key]] <- if (is.null(node_vals)) structure(list(), names = character(0)) else node_vals
}

# digits = 10 keeps node likelihoods at effectively full precision (jsonlite's
# default of 4 silently truncates them — harmless for the >=threshold age step,
# but lossy for the stored/displayed values and for strict regression checks).
write(toJSON(out, auto_unbox = TRUE, digits = 10), out_json)
