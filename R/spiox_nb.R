#' Fit a negative-binomial latent GP-IOX model
#'
#' @param Y Numeric n by q count matrix. Missing values are allowed.
#' @param X Numeric n by p covariate matrix.
#' @param coords Numeric n by 2 coordinate matrix.
#' @param offset Scalar, length-n vector, or n by q offset matrix.
#' @param r Positive length-q negative-binomial shape vector.
#' @param m Number of Vecchia neighbors.
#' @param method Currently only `"latent"`.
#' @param fit Currently only `"vi"`.
#' @param iter Maximum number of VI iterations.
#' @param print_every Progress-printing frequency.
#' @param starting List containing starting values.
#' @param opts Optional implementation controls.
#'
#' @return A fitted negative-binomial spIOX object.
#'
#' @export
spiox_nb <- function(
    Y,
    X,
    coords,
    offset = 0,
    r,
    m = 15,
    method = "latent",
    fit = "vi",
    iter = 500,
    print_every = 50,
    starting,
    opts = NULL) {
  
  # method and fit limited to latent VI for now
  method <- match.arg(method, "latent")
  fit <- match.arg(fit, "vi")
  
  if(!identical(method, "latent")) {
    stop("spiox_nb() currently supports only method = 'latent'.")
  }
  
  if(!identical(fit, "vi")) {
    stop("spiox_nb() currently supports only fit = 'vi'.")
  }
  
  # ---------------------------------------------------------------------------
  # 1. Argument Matching & Data Validation
  # ---------------------------------------------------------------------------
  # setup and dim checks
  Y <- as.matrix(Y)
  X <- as.matrix(X)
  coords <- as.matrix(coords) 
  
  n <- nrow(Y)
  q <- ncol(Y)
  p <- ncol(X)
  d <- ncol(coords)
  
  r <- as.numeric(r)
  
  observed <- !is.na(Y)
  
  stopifnot(
    "X must have same number of rows as Y" = nrow(X) == n, 
    "coords must have same number of rows as Y" = nrow(coords) == n, 
    "coords must be 2-dimensional" = d == 2L,
    "r must have length equal to ncol(Y)" = length(r) == q
  )
  
  if(any(!is.finite(r)) || any(r <= 0)) {
    stop("Every value of r must be finite and strictly positive.")
  }
  
  # checks for Y being counts (non-negative and integer)
  if(any(Y[observed] < 0)) {
    stop("Observed Y must be nonnegative.")
  }
  
  if(any(abs(Y[observed] - round(Y[observed])) > 1e-8)) {
    stop("Observed Y must be integer-valued counts.")
  }
  
  ## Helper function to reshape offset to n x q matrix
  reshape_nb_offset <- function(offset, n, q) {
    if (length(offset) == 1L) {
      # offset input a scalar
      out <- matrix(as.numeric(offset), nrow = n, ncol = q)
    } else if (is.null(dim(offset)) && length(offset) == n) {
      # offset input is a length n vector
      out <- matrix(as.numeric(offset), nrow = n, ncol = q)
    } else {
      # offset input is a n x q matrix
      out <- as.matrix(offset)
      
      if (!identical(dim(out), c(n, q))) {
        stop("offset must be a scalar, a length-n vector, or an n x q matrix.")
      }
    }
    
    out
  } # end of helper function
  
  offset <- reshape_nb_offset(offset, n, q) 
 
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
  #default_update_theta <- if (method == "latent") c(0L, 0L, 0L) else c(1L, 0L, 1L)
  
  opts_defaults <- list(
    #update_Theta      = default_update_theta,
    num_threads       = RhpcBLASctl::get_num_cores(),
    tol               = 1e-2,
    matern            = 1,
    nu                = 0.5,
    vi_pred_smp       = 0,
    # CG preconditioner choice for the latent MCMC samplers (sampling = 1 and 3).
    # One of: "auto", "jacobi", "vadu", "postcov".  "auto" (default) resolves to
    # "vadu".  Ignored for method = "response" and the single-site sampler (sampling = 2).
    cg_preconditioner = "auto" #,
    # Block latent sampler (debug$sampling = 1L) only: how B and W are drawn.
    #   TRUE  (default): joint (B, W) PCG sample (gibbs_BW_block); the B-W coupling is
    #                    preconditioned by a shared symmetric block Gauss-Seidel wrap.
    #   FALSE          : blocked route — B|W (conjugate), W|B (precision-domain PCG),
    #                    then an ASIS non-centred B refresh.  Forced FALSE for
    #                    sampling != 1 (no joint block) and method = "response".
    
    #joint_BW = TRUE,
    
    # How often the Σ/Ddiag-dependent preconditioner factors are rebuilt during MCMC.
    # A preconditioner only speeds up CG and never shifts the target, so any cadence
    # is valid; tracking the live Σ/Ddiag ("always") empirically beats freezing.
    #   "always" (default): rebuild every sweep from the live Σ/Ddiag.
    #   "auto"            : per-PC — "always" for postcov (cheap rebuild),
    #                       "once" otherwise.
    #   "once"            : build once at the autostart values, then freeze (exact).
    # NOTE: "vadu" always rebuilds regardless (its factor is a trivially cheap
    # dscale recompute).
   
     #cg_rebuild = "always" # for NB vi, omega is always updated, so always rebuild preconditioner
  )
  
  opts <- modifyList(opts_defaults, if (is.null(opts)) list() else opts)
  
  opts$vi_pred_smp <- as.integer(opts$vi_pred_smp)
  #stopifnot("opts$update_Theta must be length 3" = length(opts$update_Theta) == 3L)
  
  # Translate opts$cg_preconditioner from string to integer code expected by
  # the C++ side.  Codes (sampling = 1 block sampler and sampling = 3 per-outcome
  # sequential sampler both honour all of these):
  #   0 = auto       (resolves to vadu)
  #   1 = jacobi     (diagonal of the precision operator; manual only)
  #   2 = vadu       (Vecchia-approx-with-diagonal-update PC; Kündig & Sigrist)
  #   3 = postcov (multivariate — one block-Vecchia factor of the JOINT posterior
  #                   covariance, q×q blocks coupling all outcomes per location.
  #                   sampling=1 only; sampling=3 falls back to vadu.)
  cg_pc_codes <- c(auto = 0L, jacobi = 1L, vadu = 2L, postcov = 3L)
  cg_pc_key <- if (is.numeric(opts$cg_preconditioner)) {
    as.integer(opts$cg_preconditioner)
  } else {
    code <- cg_pc_codes[tolower(as.character(opts$cg_preconditioner))]
    if (is.na(code)) {
      stop("Invalid opts$cg_preconditioner: '",
           opts$cg_preconditioner,
           "'. Use one of: ", paste(names(cg_pc_codes), collapse = ", "), ".")
    }
    code
  }
  opts$cg_preconditioner_int <- cg_pc_key
  
  # Thread management
  if (opts$num_threads > 1) {
    RhpcBLASctl::blas_set_num_threads(1)
    message(paste0("BLAS threads set to 1, OMP threads set to ", opts$num_threads))
  } else {
    message("OMP threads set to 1.")
  }
  
  # ---------------------------------------------------------------------------
  # 4. Starting Values Initialization
  # ---------------------------------------------------------------------------
  if (missing(starting) || !is.list(starting)) {
    stop(
      "starting must be a list containing at least Theta. ",
      "Autostart for NB starting values are not yet implemented. (see spiox.R for expand)"
    )
  }
  
  `%||%` <- function(a, b) if (!is.null(a)) a else b
  
  # common vars
  Beta_start <- starting$Beta %||% matrix(0, nrow = p, ncol = q)
  Sigma_start <- starting$Sigma %||% diag(q)
  
  # latent specific var
  W_start <- starting$W %||% matrix(0, nrow = n, ncol = q)
  
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
  
  # Construct the full Theta start matrix (adding placeholder sigmasq)
  Theta_start <- rbind(
    phi     = Theta[1, , drop = FALSE],
    sigmasq = rep(1, q),                 # Internal tracking, discarded later
    nu      = Theta[2, , drop = FALSE],
    alpha   = Theta[3, , drop = FALSE]
  )
  
  # ---------------------------------------------------------------------------
  # 5. Model Dispatch
  # ---------------------------------------------------------------------------
  out <- switch(
    paste(method, fit, sep = ":"),
 
    "latent:vi" = spiox_nb_latent_vi(
      Y = Y[dag$order, , drop = FALSE],
      X = X[dag$order, , drop = FALSE],
      coords = coords[dag$order, , drop = FALSE],
      offset = offset[dag$order, , drop = FALSE],
      r = r,
      custom_dag = dag$dag,
      dag_opts = dag_opts,
      Theta = Theta_start,
      Sigma_start = Sigma_start,
      Beta_start = Beta_start,
      W_start = W_start[dag$order, , drop = FALSE],
      matern = opts$matern,
      num_threads = as.integer(opts$num_threads),
      print_every = print_every,
      tol = opts$tol,
      max_iter = iter,
      vi_pred_smp = opts$vi_pred_smp,
      cg_preconditioner = opts$cg_preconditioner_int
    ),
    
    stop("Unknown method/fit combination.")
  )
  
  # ---------------------------------------------------------------------------
  # 6. Format and Return Output
  # ---------------------------------------------------------------------------
  # Remove the internal 'sigmasq' placeholder from Theta for consistency
  if (length(dim(out$Theta)) == 2L) {
    out$Theta <- out$Theta[-2, , drop = FALSE] # vi case
  } else if (length(dim(out$Theta)) == 3L) {
    out$Theta <- out$Theta[-2, , , drop = FALSE] # mcmc case
  }
  
  # reorder offset
  inv_ord <- order(dag$order)
  if ("offset" %in% names(out)) {
    out$offset <- out$offset[inv_ord, , drop = FALSE]
  }
  
  # reorder W
  if ("W" %in% names(out)) {
    if(fit == "mcmc"){
      # an array with samples in the third dimension
      out$W <- out$W[inv_ord,,,drop=FALSE]  
    } else {
      # not an array
      out$W <- out$W[inv_ord,,drop=FALSE]
    }
  } 
  
  ## reordering the sampled Yhat
  if(anyNA(Y) && "Y_missing_indices" %in% names(out)){
    current_index <- as.integer(out$Y_missing_indices)
    
    row_reordered <- ((current_index - 1L) %% n) + 1L
    column_index <- ((current_index - 1L) %/% n) + 1L
    
    row_original <- dag$order[row_reordered]
    original_index <- row_original + (column_index - 1L) * n
    
    target_index <- which(is.na(Y))
    sample_reorder <- match(target_index, original_index)
    
    out$Y_missing_indices <- target_index
    out$Y_missing_col <- ((target_index - 1L) %/% n) + 1L
    
    if(nrow(out$Y_missing_samples) > 0L){
      out$Y_missing_samples <- out$Y_missing_samples[sample_reorder, , drop = FALSE]
    }
  }
  
  # Append metadata to the output object
  out$call     <- match.call()
  out$method   <- method
  out$fit      <- fit
  out$family <- "negative_binomial"
  
  # TRUE when the run used trustworthy autostart values: starting = "auto" or an
  # unmodified autostart() list.  FALSE for a user-built list or an autostart()
  # list with any edited element.
  # out$autostart_used <- isTRUE(starting_is_autostart)
  
  out$gridded  <- gridded
  out$dag_opts <- dag_opts
  out$dag_info <- dag
  out$coords   <- coords
  
  return(out)
}