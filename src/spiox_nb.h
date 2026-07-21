#pragma once

#include "spiox.h"

// This is the src/spiox_nb.h (header file) for the spiox_nb

class SpIOXNB {
public:
  // having the SpIOX class as core so we have everything in it
  // we dont have Ddiag so put in arma::ones(q) but ignore it
  SpIOX core;
  
  //  additional NB-specific stuff
  arma::vec r;
  arma::mat offset;
  // c_ij = offset_ij - log r_j
  arma::mat c;
  // kappa_ij = (y_ij - r_j)/2
  arma::mat kappa;
  // omega_ij = z_ij (mcmc) or E_q[z_ij] (vi)
  arma::mat omega;
  arma::mat Epsi2;
  
  // Sigma psi, mean, inverse, sqrt of inverse
  arma::mat Sigma_scale;
  arma::mat Sigma_mean;
  arma::mat Qbar;
  arma::mat Qbar_sqrt;
  
  arma::mat VTV_ma;
  
  // NB VI methods
  void update_pg_expectation();
  
  void update_sigma_nbvi();
  
  void nb_latent_vi(SpIOX::PrecondChoice preconditioner);
  
  // Only compute and evaluate the ll without changing anything in the model
  double nb_latent_fit_eval() const;
  
  // function for misalignment prediction
  arma::vec nb_latent_vi_smp4Yhat() const;
  
  // Constructor
  SpIOXNB(const arma::mat& Y,
          const arma::mat& X,
          const arma::mat& coords, 
          const arma::field<arma::uvec>& custom_dag,
          int dag_opts,
          const arma::mat& Theta,
          const arma::mat& Beta_start,
          const arma::mat& W_start,
          const arma::mat& Sigma_start,
          const arma::vec& r_in,
          const arma::mat& offset_in,
          int matern, 
          int num_threads, 
          int vi_min_iter)
    :
    core(
      Y, X, coords, custom_dag, dag_opts, 1,
      Beta_start, W_start, Sigma_start, Theta,
      // don't update any theta
      arma::zeros<arma::uvec>(4),
      // dummy Ddiag (but we dont use it)
      arma::ones<arma::vec>(Y.n_cols), 
      matern, num_threads, vi_min_iter
    ),
    r(r_in),
    offset(offset_in),
    VTV_ma(arma::zeros<arma::mat>(Y.n_cols, Y.n_cols))
  {
    // start of constructor code
    const arma::uword n = core.n;
    const arma::uword q = core.q;
    
    const arma::mat r_mat = arma::repmat(r.t(), n, 1);
    const arma::mat log_r_mat = arma::repmat(arma::log(r).t(), n, 1);
    c = offset - log_r_mat;
    kappa = 0.5 * (core.Y - r_mat);
    
    // Missing observations contribute no likelihood
    const arma::uvec missing = arma::find(core.missing_mat == 1);
     
    kappa.elem(missing).zeros(); 
    c.elem(missing).zeros();
    
    // initializing the Psi, SIgma, Q, W
    const arma::mat psi0 = core.X * core.B + core.W + c;
    Epsi2 = arma::square(psi0); 
    
    // Initialize omega using Psi0
    omega.zeros(n, q);
    update_pg_expectation();
    
    Sigma_mean = Sigma_start;
    Qbar = arma::inv_sympd(arma::symmatu(Sigma_start));
    Qbar_sqrt = arma::chol(arma::symmatu(Qbar), "lower");
    
    // Initialize VTV from W_start
    core.compute_V();
    VTV_ma = core.V.t() * core.V;
    // prior for Sigma scale = arma::eye<arma::mat>(q, q)
    Sigma_scale = arma::eye<arma::mat>(q, q) + VTV_ma;
  }

};