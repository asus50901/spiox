autostart <- function(Y, X, coords, method=c("response", "latent"), m=15, nu=0.5, verbose=TRUE){
  
  method <- match.arg(method)
  
  p <- ncol(X)
  q <- ncol(Y)
  n <- nrow(Y)
  
  Beta <- matrix(0, nrow=p, ncol=q)
  Theta <- matrix(0, nrow=3, ncol=q)
  Sigma <- diag(q)
  
  if(method == "latent"){
    Ddiag <- rep(0, q)
    W <- matrix(0, nrow=n, ncol=q)
    
    # fill missing (just here)
    Yfill <- Y
    if(any(is.na(Y))){
      for(j in seq_len(q)){
        ix_na <- is.na(Yfill[,j])
        n_na <- sum(ix_na)
        mu_j = mean(Yfill[,j], na.rm=TRUE)
        sd_j = sd(Yfill[,j], na.rm=TRUE)
        Yfill[ix_na, j] <- rnorm(n_na, mean = mu_j, sd = sd_j)
      }  
    }
  } else {
    W <- matrix(0, 1, 1)
  }
  
  if(verbose) cat("Processing ", q, " outcomes... [ ")
  
  if(nu==0.5){
    covfun_name = "exponential_isotropic"
  } else if(nu==1.5) {
    covfun_name = "matern15_isotropic"
  } else {
    covfun_name = "matern_isotropic"
    message("nu smoothness value != c(0.5, 1.5), fitting matern_isotropic\n")
  }
  
  for(j in seq_len(q)){
    if(verbose) cat(j, " ")
    y_j  <- Y[, j]
    invisible(capture.output({out  <- GpGp::fit_model(y = y_j, locs = coords,
                                             X = X, m_seq = m, 
                                             covfun_name = covfun_name,
                                             silent = TRUE)}))
    # nugget = sigma * tsq
    # spatial variance = sigma
    # total = sigma (1+tsq)
    # alpha (portion of meas error): tsq/(1+tsq)
    if(covfun_name == "matern_isotropic"){
      sigmasq <- out$covparms[1]
      tsq     <- out$covparms[4]
      alpha   <- tsq / (1+tsq)
      phi     <- 1 / out$covparms[2]
      nu      <- out$covparms[3]
    } else {
      sigmasq <- out$covparms[1]
      tsq     <- out$covparms[3]
      alpha   <- tsq / (1+tsq)
      phi     <- 1 / out$covparms[2]
      nu      <- nu
    }
    
    Beta[,j] <- out$betahat
    if(method == "latent"){
      W[,j] <- Yfill[,j] - X %*% out$betahat
      Sigma[j,j] <- sigmasq
      Ddiag[j] <- sigmasq * tsq
      Theta[,j] <- c(phi, nu, 0)
    } else {
      Sigma[j,j] <- sigmasq * (1+tsq)
      Theta[,j] <- c(phi, nu, alpha)
    }
  }
  
  if(verbose) cat("]\n")

  if(method == "response"){
    out <- list(
      Beta = Beta,
      Sigma = Sigma,
      Theta = Theta
    )
  } else {
    out <- list(
      Beta = Beta,
      W = W,
      Sigma = Sigma,
      Theta = Theta,
      Ddiag = Ddiag
    )
  }

  # Tamper-evident autostart marker.  We stash a reference copy of the freshly
  # produced list as an attribute (the copy itself carries no such attribute).
  # spiox() can then tell three cases apart by comparing the list against this
  # reference with identical():
  #   - attribute absent          -> user-built list (warn: PC built on them)
  #   - present and identical      -> untouched autostart values (trusted)
  #   - present but not identical  -> user edited some element(s) (warn)
  # The attribute survives `$<-` / `[[<-` element edits, so any manual change to
  # Theta/Sigma/W/Ddiag/Beta is detected.
  attr(out, "spiox_autostart_ref") <- out

  return(out)


}


## autostart for spiox_nb() 
# we use the obs data to fit uni model instead of imputing the loss ones. 
# just an easy fix.
NB_autostart <- function(X, Y, coords,
                         m, likelihood, cov_function = "matern_estimate_shape",
                         matern_nu = NULL,
                         method = "mle", offset = NULL,
                         boost_opts = list(),
                         seed, num_threads){
  # dim stuff
  n <- nrow(Y)
  q <- ncol(Y)
  p <- ncol(X)
  
  # handling misalignment (use only the observed data to fit uni model)
  n_obs <- colSums(!is.na(Y))
  
  cov_function <- match.arg(cov_function, choices = c("matern_estimate_shape", "matern", "exponential", "gaussian"))
  method <- match.arg(method, choices = c("mle", "boost"))
  
  # special handling for matern (with nu fixed)
  if (cov_function == "matern") {
    
    # stopping triggers
    stopifnot("`matern_nu` must be supplied when cov_function = 'matern'." = !is.null(matern_nu),
              "`matern_nu` must have length 1 or length q." = length(matern_nu) %in% c(1, q),
              "All values of `matern_nu` must be positive." = all(matern_nu > 0))
    
    # given a single number
    if (length(matern_nu) == 1) matern_nu <- rep(matern_nu, q)
    
  } else if (!is.null(matern_nu)) {
    
    warning("`matern_nu` is only used when cov_function = 'matern'. ",
            "Ignored for cov_function = '", cov_function, "'.")
  }
  
  if (is.null(offset)) {
    offset <- matrix(1, nrow = n, ncol = q) # log(1) = 0
  } else {
    
    if (!is.matrix(offset)) stop("`offset` must be NULL or a matrix of n x q")
    
    observed_offset <- !is.na(Y)# only need offset for the obs. Y
  }
  
  # Handing boost opts
  default_boost_opts <- list(nrounds = 100L, learning_rate = 0.05,
                             min_data_in_leaf = 20L, max_depth = NULL, verbose = 1L)
  
  unknown_boost_opts <- setdiff( names(boost_opts), names(default_boost_opts))
  boost_opts <- utils::modifyList(default_boost_opts, boost_opts)
  
  # Helper function - Convert GPBoost covariance parameters to spIOX parameterization 
  gpboost_to_spiox <- function(cov_pars, cov_function, cov_fct_shape = NULL) {
    
    # get_cov_pars(std_err = FALSE) should return a named numeric vector.
    # This also gives a matrix if std_err = TRUE is ever used later.
    if (is.matrix(cov_pars)) {
      
      if ("Param." %in% rownames(cov_pars)) {
        cov_pars <- cov_pars["Param.", , drop = TRUE]
      } else {
        cov_pars <- cov_pars[1, , drop = TRUE]
      }
    } 
    
    required <- c("GP_var", "GP_range") 
    
    marginal_variance <- unname(cov_pars[["GP_var"]])
    gp_range <- unname(cov_pars[["GP_range"]]) 
    
    ## Case 1 - Matern with nu estimated
    if (cov_function == "matern_estimate_shape") { 
      
      nu <- unname(cov_pars[["GP_smoothness"]])
      phi <- sqrt(2 * nu) / gp_range # following the Williams and Rasmussen (2006) parameterization
      spiox_matern <- 1
    }
    
    ## Case 2 - Matern (nu fixed at some value)
    else if (cov_function == "matern") {
      
      nu <- cov_fct_shape
      phi <- sqrt(2 * nu) / gp_range
      spiox_matern <- 1
    }
    
    # Case 3 - Exponential
    #
    # GPBoost gives exp(-d / range)
    # spIOX needs power exponential => exp(-phi * d^nu)
    #
    # So, we have nu = 1 and  phi = 1 / range 
    else if (cov_function == "exponential") {
      
      nu <- 1 # nu here is not the spatial smoothness of Matern, but the exponent for the power-expo in spIOX
      phi <- 1 / gp_range
      spiox_matern <- 0
    }
    
    # Case 4 - Gaussian
    #
    # GPBoost gives exp(-(d / range)^2)
    # spIOX needs power exponential => exp(-phi * d^nu)
    #
    # So, we have  nu = 2 and phi = 1 / range^2 
    else if (cov_function == "gaussian") {
      
      nu <- 2
      phi <- 1 / gp_range^2
      spiox_matern <- 0L
    }
    else {
      stop("Unsupported covariance function.")
    }
    
    return(c(marg_Sigma = marginal_variance,
             phi = phi, nu = nu,
             gp_range = gp_range,
             spiox_matern = spiox_matern))
  }
  
  # storing 
  gpboost_model <- vector("list", q)
  cov_mod_GPM <- vector("list", q)
  timing <- numeric(q)
  
  res_theta <- matrix(NA_real_, nrow = q, ncol = 3,
                      dimnames = list(NULL, c("marg_Sigma", "phi", "nu")))
  res_auxVar <- vector("list", q)
  res_beta <- matrix(NA_real_, nrow = p, ncol = q)
  
  gp_range_est <- numeric(q)
  
  observed_index <- vector("list", q)
  n_obs <- integer(q)
  
  # keep the naming
  outcome_names <- colnames(Y)
  
  if (!is.null(outcome_names)) {
    names(gpboost_model) <- outcome_names
    names(cov_mod_GPM) <- outcome_names
    names(res_auxVar) <- outcome_names
    names(timing) <- outcome_names
    colnames(res_beta) <- outcome_names
    rownames(res_theta) <- outcome_names
    names(gp_range_est) <- outcome_names
  }
  
  rownames(res_beta) <- colnames(X)
  
  spiox_matern <- if (cov_function %in% c("matern_estimate_shape", "matern")) { 1L } else { 0L }
  
  # Actual fitting model
  for(i in 1:q){
    
    # handling misalignment
    obs_i <- !is.na(Y[, i])
    observed_index[[i]] <- which(obs_i)
    n_obs[i] <- sum(obs_i)
    
    # Outcome-specific data
    Y_i <- as.numeric(Y[obs_i, i])
    X_i <- X[obs_i, ,drop = FALSE]
    coords_i <- coords[obs_i, , drop = FALSE]
    offset_i <- offset[obs_i, i]
    
    # GPModel fit
    gp_args <- list(gp_coords = coords_i,
                    gp_approx = "vecchia_latent", num_neighbors = as.integer(m),
                    cov_function = cov_function, likelihood = likelihood,
                    seed = seed, GPU_use = FALSE, num_parallel_threads = as.integer(num_threads))
    
    # stuff for matern with nu fixed
    nu_i <- if(cov_function == "matern"){ matern_nu[i] } else { NULL }
    if (cov_function == "matern") { gp_args$cov_fct_shape <- nu_i }
    
    gp_model <- do.call(gpboost::GPModel, gp_args)
    
    # fiting the MLE or Boosting model
    if(method == "mle"){
      # fitting via spatial GLMM using Laplace approximation
      # This gives Beta est.
      cat("\nFitting Outcome", i, " via Laplace approximation\n")
      
      gp_model$set_optim_params(params = list(optimizer_cov = "lbfgs", delta_rel_conv = 1e-6, maxit = 3000))
      
      tmp_timing <- system.time({
        gp_model$fit(y = Y_i,
                     X = X_i,
                     offset = log(offset_i))
      })
      
      cov_mod_GPM[[i]] <- gp_model
      
      # tidying Beta
      res_beta[, i] <- cov_mod_GPM[[i]]$get_coef() # Beta
      
    } else {
      # fitting via boosting
      cat("\nFitting Outcome", i, " via boosting\n")
      
      train_dataset <- gpboost::gpb.Dataset(data = X_i, label = Y_i)
      
      # setting the offset term
      gpboost::setinfo(train_dataset, "init_score", log(offset_i))
      
      # Direct gpboost() arguments
      nrounds <- as.integer(boost_opts$nrounds)
      verbose <- boost_opts$verbose
      
      # Everything else is passed through `params`
      boost_params <- boost_opts[setdiff(names(boost_opts), c("nrounds", "verbose"))]
      
      # Don't pass parameters with NULL values.
      boost_params <- Filter(f = Negate(is.null), x = boost_params)
      
      # Main function arguments take precedence.
      boost_params$num_threads <- as.integer(num_threads)
      boost_params$seed <- seed
      
      tmp_timing <- system.time({
        gpboost_model[[i]] <- gpboost::gpboost(data = train_dataset,
                                               gp_model = gp_model,
                                               params = boost_params,
                                               nrounds = nrounds, 
                                               verbose = verbose)
      })
      
      cov_mod_GPM[[i]] <- gp_model
    }
    
    timing[i] <- unname(tmp_timing[[3]]) # by seconds
    
    # Tidying and post-processing
    # follows the Matern parameterization of Rasmussen and Williams (2006)
    tmp_cov_pars <- cov_mod_GPM[[i]]$get_cov_pars() # marginal variance (sigma_ii), spatial range = sqrt(2*nu)/phi and nu
    
    tmp_spiox <- gpboost_to_spiox(cov_pars = tmp_cov_pars, cov_function = cov_function, cov_fct_shape = nu_i)
    
    res_theta[i, ] <- tmp_spiox[c("marg_Sigma", "phi", "nu")]
    gp_range_est[i] <- tmp_spiox[["gp_range"]]
    res_auxVar[[i]] <- cov_mod_GPM[[i]]$get_aux_pars() # overdispersion_R est.
  }
  
  # ready for output
  if (cov_function %in% c("matern_estimate_shape", "matern")) {
    
    Theta_start <- rbind(phi = res_theta[, "phi"],
                         nu = res_theta[, "nu"],
                         alpha = rep(0, q))
  } else {
    Theta_start <- rbind(phi = res_theta[, "phi"],
                         pwExpo_power = res_theta[, "nu"],
                         alpha = rep(0, q))
  }
  
  if (!is.null(outcome_names)) { colnames(Theta_start) <- outcome_names }
  
  # changes for the naming
  if (cov_function %in% c("matern_estimate_shape", "matern")) {
    colnames(res_theta)[3] <- "nu"
  } else {
    colnames(res_theta)[3] <- "pwExpo_exponent"
  }
  
  if (method == "boost") { res_beta <- NULL }
  
  
  return(list(beta_est = res_beta,
              theta_est = res_theta,
              Theta_start = Theta_start,
              spiox_matern = spiox_matern,
              aux_param_est = res_auxVar,
              time = timing,
              # for misalignment
              n_obs = n_obs,
              observed_index = observed_index,
              likelihood = likelihood,
              cov_function = cov_function,
              matern_nu = matern_nu,
              method = method, Vecchia_NN = m,
              gpboost_models = cov_mod_GPM))
}