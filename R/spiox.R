#' Fit GP-IOX models for multivariate spatial outcomes
#'
#' @description
#' Fits the GP-IOX response or latent model using either Markov Chain Monte Carlo
#' (MCMC) or Variational Inference (VI). A Vecchia Directed Acyclic Graph (DAG) is 
#' constructed internally from `coords` and `m` (grid-aware if `coords` are gridded).
#'
#' @param Y Numeric matrix of responses with dimensions `n x q` (rows = locations, columns = outcomes).
#' @param X Numeric design matrix with dimensions `n x p` (must have the same number of rows as `Y`).
#' @param coords Numeric matrix of spatial coordinates with dimensions `n x 2` (must have the same number of rows as `Y`).
#' @param m Integer. Vecchia neighbor size used to construct the DAG. Defaults to 15.
#' @param method Character. One of `"response"` or `"latent"`.
#' @param fit Character. One of `"mcmc"` or `"vi"`.
#' @param iter Integer. Number of MCMC iterations if `fit="mcmc"`. If `fit="vi"`, this is the maximum number of iterations allowed.
#' @param print_every Integer. Print progress every `print_every` iterations. Set to `0` for silent.
#' @param starting Character (`"auto"`) or named list of starting values. Any omitted elements are initialized internally. Theta requires care:
#'   \itemize{
#'     \item `Beta`: numeric matrix `p x q` (starting regression coefficients, defaults to all `0`).
#'     \item `Sigma`: numeric matrix `q x q` (starting outcome covariance / coregionalization, defaults to `diag(q)`).
#'     \item `Theta`: numeric matrix `3 x q` with rows corresponding to `phi`, `nu`, `alpha`.
#'       `Theta` must be supplied when `fit="vi"` or when `opts$update_Theta = c(0,0,0)`.
#'     \item `Ddiag`: numeric vector of length `q` (latent-model only; defaults to `rep(1, q)`).
#'   }
#' @param opts Optional named list of algorithm/model options. Any omitted elements fall back to defaults.
#'   \itemize{
#'     \item `matern`: integer (0=squared exponential (nu is the exponent); 1=Matern (nu is smoothness)).
#'     \item `num_threads`: integer number of threads (OpenMP if available, defaults to `1`).
#'     \item `update_Theta`: integer vector of length 3 indicating which of (`phi`, `nu`, `alpha`) to update via MCMC.
#'       Defaults: latent MCMC `c(0,0,0)`; response MCMC `c(1,0,1)`. Ignored for `fit='vi'`.
#'     \item `tol`: numeric tolerance used for VI stopping.
#'   }
#' @param debug Optional named list of MCMC controls (primarily for development). Any omitted elements fall back to defaults.
#'   \itemize{
#'     \item `sampling`: integer specifying the latent-model sampler variant (only used when `method="latent"` and `fit="mcmc"`).
#'     \item `sample_Beta`: logical; sample regression coefficients in MCMC.
#'     \item `sample_Sigma`: logical; sample Sigma in MCMC.
#'     \item `sample_Ddiag`: logical; sample D in latent-model MCMC.
#'   }
#'
#' @return A list containing the fitted model parameters (e.g., `Theta`, `Beta`), the original `call`, `method`, `fit`, the `gridded` flag, and `dag_opts`.
#'
#' @examples
#' \dontrun{
#' # Example: Fit a GP-IOX response model using Variational Inference (VI)
#' 
#' fit_time <- system.time({
#'   spiox_out <- spiox(
#'     Y = Y, 
#'     X = X, 
#'     coords = coords, 
#'     m = 15, 
#'     method = "response", 
#'     fit = "vi", 
#'     iter = 200, 
#'     print_every = 100
#'   )
#' })
#' }
#' @export
spiox <- function(Y, X, coords, m = 15, 
                  method = c("response", "latent"),
                  fit = c("mcmc", "vi"),
                  iter = 1000, print_every = 100, 
                  starting = "auto", opts = NULL, debug = NULL) {
  
  # ---------------------------------------------------------------------------
  # 1. Argument Matching & Data Validation
  # ---------------------------------------------------------------------------
  method <- match.arg(method)
  fit    <- match.arg(fit)
  
  Y <- as.matrix(Y)
  X <- as.matrix(X)
  coords <- as.matrix(coords)
  
  if( (fit == "vi") & (method == "response") ){
    cc <- complete.cases(Y)
    
    Y <- Y[cc,,drop=FALSE]
    X <- X[cc,,drop=FALSE]
    coords <- coords[cc,,drop=FALSE]
    
    nava <- sum(cc)
    warning(paste0("Fitting on ", nava, " points (others are partly missing)"))
  }
  
  n <- nrow(Y)
  q <- ncol(Y)
  p <- ncol(X)
  d <- ncol(coords)
  
  stopifnot(
    "X must have same number of rows as Y" = nrow(X) == n, 
    "coords must have same number of rows as Y" = nrow(coords) == n, 
    "coords must be 2-dimensional" = d == 2L
  )
  
  # ---------------------------------------------------------------------------
  # 2. Spatial Grid Evaluation & Vecchia DAG Construction
  # ---------------------------------------------------------------------------
  # Helper to check if coordinates fall on a regular grid
  is_gridded <- function(xy, tol = sqrt(.Machine$double.eps)) {
    x <- sort(unique(xy[, 1]))
    y <- sort(unique(xy[, 2]))
    dx <- diff(x)
    dy <- diff(y)
    
    ok_step <- function(d) {
      length(d) <= 1L || 
        max(d, na.rm = TRUE) <= tol || 
        (max(d) - min(d) <= tol * max(1, max(d)))
    }
    
    nx <- length(x)
    ny <- length(y)
    
    (nrow(unique(xy)) == nx * ny) && ok_step(dx) && ok_step(dy)
  }
  
  gridded <- is_gridded(coords)
  dag <- dag_vecchia_o(coords, m, gridded)
  dag_opts <- if (gridded) -1L else 0L
  
  # ---------------------------------------------------------------------------
  # 3. Model & Debug Options Setup
  # ---------------------------------------------------------------------------
  # Set method-specific defaults for Theta updates
  default_update_theta <- if (method == "latent") c(0L, 0L, 0L) else c(1L, 0L, 1L)
  
  opts_defaults <- list(
    update_Theta = default_update_theta,
    num_threads  = RhpcBLASctl::get_num_cores(),
    tol          = 1e-2,
    matern       = 1,
    nu           = 0.5,
    vi_pred_smp  = 0
  )
  opts <- modifyList(opts_defaults, if (is.null(opts)) list() else opts)
  opts$vi_pred_smp <- as.integer(opts$vi_pred_smp)
  stopifnot("opts$update_Theta must be length 3" = length(opts$update_Theta) == 3L)
  
  # Thread management
  if (opts$num_threads > 1) {
    RhpcBLASctl::blas_set_num_threads(1)
    message(paste0("BLAS threads set to 1, OMP threads set to ", opts$num_threads))
  } else {
    message("OMP threads set to 1.")
  }
  
  # Debug controls (primarily for MCMC development)
  debug_defaults <- list(
    sampling     = 2L,  # 2 = n sequential, q block
    sample_Beta  = TRUE,
    sample_Sigma = TRUE,
    sample_Ddiag = TRUE
  )
  debug <- modifyList(debug_defaults, if (is.null(debug)) list() else debug)
  
  # ---------------------------------------------------------------------------
  # 4. Starting Values Initialization
  # ---------------------------------------------------------------------------
  auto_do <- is.character(starting) && length(starting) == 1 && starting == "auto"
  
  if (!auto_do && !is.list(starting)) {
    stop("Invalid input for 'starting'. Specify 'auto' or provide a list.")
  }
  
  if (auto_do) {
    # Generate automatic starting values
    starting <- autostart(Y, X, coords, method, m = m, opts$nu)
    Beta_start  <- starting$Beta
    Sigma_start <- starting$Sigma
    Theta       <- starting$Theta
    
    if (method == "latent") {
      W_start     <- starting$W
      Ddiag_start <- starting$Ddiag
    }
  } else {
    # Process user-provided starting values
    `%||%` <- function(a, b) if (!is.null(a)) a else b
    
    # Common variables
    Beta_start  <- starting$Beta  %||% matrix(0, nrow = p, ncol = q)
    Sigma_start <- starting$Sigma %||% diag(q)
    
    # Latent-only variables
    if (method == "latent") {
      W_start     <- starting$W     %||% matrix(0, nrow = n, ncol = q)
      Ddiag_start <- starting$Ddiag %||% diag(q)
    } 
    
    # Validate Theta based on user's fit/update options
    if (all(opts$update_Theta == 0L) && is.null(starting$Theta)) {
      stop("starting$Theta must be supplied when update_Theta = c(0,0,0).")
    }
    if (fit == "vi" && is.null(starting$Theta)) {
      stop("starting$Theta (3 x q with rows phi, nu, alpha) must be supplied for fit='vi'.")
    }
    
    Theta <- starting$Theta %||% NULL
    
    if (!is.null(Theta)) {
      Theta <- as.matrix(Theta)
      stopifnot("Theta must be 3 x q" = nrow(Theta) == 3L, ncol(Theta) == q)
    } else {
      # Fallback defaults if Theta is partially updated
      alpha_default <- ifelse(method == "latent", 0, 0.1)
      Theta <- rbind(10, 0.5, alpha_default)
    }
    rownames(Theta) <- c("phi", "nu", "alpha")
    
    # Alert user if they passed invalid alpha for latent method
    if (method == "latent" && !is.null(Theta)) {
      alpha <- as.numeric(Theta[3, ])
      if (any(abs(alpha) > 0)) {
        warning("In method='latent', alpha (last row of starting$Theta) should be 0; nonzero values were supplied.")
      }
    }
  }
  
  # Construct the full Theta start matrix (adding placeholder sigmasq)
  Theta_start <- rbind(
    phi     = Theta[1, , drop = FALSE],
    sigmasq = rep(1, q),                 # Internal tracking, discarded later
    nu      = Theta[2, , drop = FALSE],
    alpha   = Theta[3, , drop = FALSE]
  )
  
  # Format the update bools (phi, sigmasq (fixed 0), nu, alpha)
  update_Theta_full <- as.integer(
    c(opts$update_Theta[1], 0L, opts$update_Theta[2], opts$update_Theta[3])
  )
  if(method == "latent"){
    if(opts$update_Theta[3] == 1){
      update_Theta_full[4] <- 0
      warning("MCMC for alpha disabled because method = 'latent' (opts$update_Theta[3] forced to 0)")
    }
  }
  
  # ---------------------------------------------------------------------------
  # 5. Model Dispatch
  # ---------------------------------------------------------------------------
  out <- switch(
    paste(method, fit, sep = ":"),
    
    "response:mcmc" = spiox_response(
      Y[dag$order,,drop=F], X[dag$order,,drop=F], coords[dag$order,,drop=F], 
      dag$dag,
      Beta_start, Sigma_start, Theta_start,
      mcmc         = iter, 
      print_every  = print_every, 
      matern       = opts$matern, 
      dag_opts     = dag_opts,
      sample_Beta  = debug$sample_Beta, 
      sample_Sigma = debug$sample_Sigma,
      update_Theta = update_Theta_full,
      num_threads  = as.integer(opts$num_threads)
    ),
    
    "latent:mcmc" = spiox_latent(
      Y[dag$order,,drop=F], X[dag$order,,drop=F], coords[dag$order,,drop=F], 
      dag$dag,
      Beta_start, 
      W_start[dag$order,,drop=F], 
      Sigma_start, Theta_start, Ddiag_start,
      mcmc         = iter, 
      print_every  = print_every, 
      matern       = opts$matern, 
      dag_opts     = dag_opts,
      sample_Beta  = debug$sample_Beta, 
      sample_Sigma = debug$sample_Sigma,
      sample_Ddiag = debug$sample_Ddiag,
      update_Theta = update_Theta_full,
      num_threads  = as.integer(opts$num_threads),
      sampling     = as.integer(debug$sampling)
    ),
    
    "response:vi" = spiox_response_vi(
      Y[dag$order,,drop=F], X[dag$order,,drop=F], coords[dag$order,,drop=F],  
      dag$dag, dag_opts,
      Theta        = Theta_start, 
      Sigma_start  = Sigma_start, 
      Beta_start   = Beta_start,
      print_every  = print_every, 
      matern       = opts$matern,
      num_threads  = as.integer(opts$num_threads)
    ),
    
    "latent:vi" = spiox_latent_vi(
      Y[dag$order,,drop=F], X[dag$order,,drop=F], coords[dag$order,,drop=F], 
      dag$dag, dag_opts,
      Theta        = Theta_start, 
      Sigma_start  = Sigma_start, 
      Beta_start   = Beta_start,
      W_start      = W_start[dag$order,,drop=F],
      Ddiag_start  = Ddiag_start,
      matern       = opts$matern, 
      num_threads  = as.integer(opts$num_threads),
      print_every  = print_every, 
      tol          = opts$tol, 
      max_iter     = iter,
      vi_pred_smp = opts$vi_pred_smp
    ),
    
    stop("Unknown method/fit combination.")
  )
  
  # ---------------------------------------------------------------------------
  # 6. Format and Return Output
  # ---------------------------------------------------------------------------
  # Remove the internal 'sigmasq' placeholder from Theta for consistency
  if ("Theta" %in% names(out)) {
    out$Theta <- out$Theta[-2, , , drop = FALSE]
  } else {
    out$Theta <- Theta_start[-2, , drop = FALSE]
  }
  
  # reorder W
  if ("W" %in% names(out)) {
    if(fit == "mcmc"){
      # an array with samples in the third dimension
      out$W <- out$W[order(dag$order),,,drop=FALSE]  
    } else {
      # not an array
      out$W <- out$W[order(dag$order),,drop=FALSE]
    }
  } 
  
  # indices of imputed, response:mcmc
  if ( (method == "response") & (fit == "mcmc") & any(is.na(Y)) ) {
    ord <- dag$order
    cur <- which(is.na(Y[ord, ]))
    tgt <- as.numeric(which(is.na(Y)))
    
    # map current indices to original-space indices
    n <- nrow(Y)
    row_reord <- ((cur - 1) %% n) + 1
    col_idx   <- ((cur - 1) %/% n) + 1
    row_orig  <- ord[row_reord]
    orig_idx  <- row_orig + (col_idx - 1) * n
    
    # orig_idx is a permutation of tgt; find the reordering
    reorder <- match(tgt, orig_idx)
    
    out$Y_missing_indices <- tgt
    out$Y_missing_samples <- out$Y_missing_samples[reorder, , drop=FALSE]
    out$Y_missing_col <- ((tgt - 1) %/% n) + 1
  }
  
  # Append metadata to the output object
  out$call     <- match.call()
  out$method   <- method
  out$fit      <- fit
  out$gridded  <- gridded
  out$dag_opts <- dag_opts
  out$dag_info <- dag
  out$coords   <- coords
  
  return(out)
}