# Call the unmodified coMDS package on embeddings written by comds_bridge.py.
# Arguments:
#   1 payload file
#   2 output directory
#   3 "comds" or "locomds"
#   4 tau (LoCoMDS only)
#   5 percentile (LoCoMDS only)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) {
  stop("Usage: fit_consensus.R payload outdir comds|locomds [tau percentile]")
}

payload_path <- args[[1]]
out_dir <- args[[2]]
kind <- args[[3]]
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

read_payload <- function(path) {
  con <- file(path, "rb")
  on.exit(close(con), add = TRUE)
  hdr <- readBin(con, integer(), n = 4)
  n <- hdr[[1]]
  d <- hdr[[2]]
  m <- hdr[[3]]
  has_w <- hdr[[4]]
  emb <- vector("list", m)
  for (j in seq_len(m)) {
    raw <- readBin(con, double(), n = n * d)
    emb[[j]] <- matrix(raw, nrow = n, ncol = d)
  }
  weights <- NULL
  if (has_w == 1L) {
    weights <- readBin(con, double(), n = m)
  }
  has_wm <- readBin(con, integer(), n = 1)
  weightmat <- NULL
  if (length(has_wm) == 1 && has_wm == 1L) {
    weightmat <- vector("list", m)
    for (j in seq_len(m)) {
      raw <- readBin(con, double(), n = n * n)
      weightmat[[j]] <- matrix(raw, nrow = n, ncol = n)
    }
  }
  list(emb = emb, weights = weights, weightmat = weightmat, n = n, d = d, m = m)
}

payload <- read_payload(payload_path)
names(payload$emb) <- sprintf("view%03d", seq_len(payload$m))

library(coMDS)

t0 <- proc.time()[["elapsed"]]
fit <- if (kind == "comds") {
  coMDS(
    embed_list = payload$emb,
    ndim = 2,
    constraint = "indscal",
    weights = payload$weights,
    weightmat = payload$weightmat,
    itmax = 300,
    eps = 1e-6,
    verbose = FALSE,
    weightlam = if (is.null(payload$weightmat)) 0 else 1e-6
  )
} else if (kind == "locomds") {
  tau <- as.numeric(args[[4]])
  percentile <- as.numeric(args[[5]])
  locoMDS(
    embed_list = payload$emb,
    ndim = 2,
    tau = tau,
    percentile = percentile,
    constraint = "indscal",
    weights = payload$weights,
    itmax = 300,
    eps = 1e-6,
    verbose = FALSE,
    weightlam = 1e-6
  )
} else {
  stop("Unknown kind: ", kind)
}
elapsed <- proc.time()[["elapsed"]] - t0

gspace <- as.matrix(fit$gspace)
write.table(
  gspace,
  file = file.path(out_dir, "gspace.csv"),
  sep = ",",
  row.names = FALSE,
  col.names = FALSE
)

sps <- if (!is.null(fit$sps)) as.numeric(fit$sps) else rep(NA_real_, payload$m)
write.table(
  data.frame(view = names(payload$emb), sps = sps),
  file = file.path(out_dir, "sps.csv"),
  sep = ",",
  row.names = FALSE,
  col.names = TRUE
)

meta <- c(
  paste0("stress=", format(fit$stress, digits = 16, scientific = TRUE)),
  paste0("rss=", format(fit$rss, digits = 16, scientific = TRUE)),
  paste0("niter=", as.integer(fit$niter)),
  paste0("seconds=", format(elapsed, digits = 16, scientific = TRUE)),
  paste0("nobj=", as.integer(fit$nobj)),
  paste0("model=", fit$model)
)
writeLines(meta, file.path(out_dir, "meta.txt"))
cat("ok stress=", fit$stress, " niter=", fit$niter, " seconds=", elapsed, "\n", sep = "")
