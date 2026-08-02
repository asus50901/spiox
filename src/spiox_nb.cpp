#include "spiox_nb.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>

// This file is for the PolyaGamma expectation, NB vi iteration,
// Sigma update, likelihood calc, and making predictions
// put some lower level functions that the NB model calculations here. 

// Helper functions to compute the NB likelihood (only usable within this cpp file)
namespace{
// compute log(exp(a) + exp(b))
// by ( m + log{exp(a - m) + exp(b - m)} ) where m = max(a, b) 
inline double logaddexp_comp(const double a, const double b){
  const double m = std::max(a, b);
  return m + std::log(std::exp(a - m) + std::exp(b - m));
}
  
// Computing the NegBin prob
inline double logistic_comp(const double x){
  if(x >= 0.0) return 1.0 / (1.0 + std::exp(-x));
  
  const double exp_x = std::exp(x);
  return exp_x / (1.0 + exp_x);
}
  
} // end of anonymous namespace

// Function to update Eq[z_ij] for the Polya-Gamma augmented variable
void SpIOXNB::update_pg_expectation(){
  const arma::uword n = core.n;
  const arma::uword q = core.q;
  
  for(arma::uword j = 0; j < q; ++j){
    for(arma::uword i = 0; i < n; ++i){
      
      if(core.missing_mat(i, j)){
        omega(i, j) = 0.0;
        continue;
      }
      
      // b_ij = y_ij + r_j
      const double bij = core.Y(i, j) + r(j);
      
      // Build xi2 = Epsi2, but lower cap at 0
      const double xi2 = std::max(Epsi2(i, j), 0.0);
      double omega_ij;
      
      // Compute the Eq[z_ij] = b * [1/4 - xi^2/48 + xi^4/480] if xi2 small
      // Otherwise compute it directly in Polson et al. form
      if(xi2 < 1e-8){
        omega_ij = bij * (0.25 - xi2 / 48.0 + xi2 * xi2 / 480.0);
      } else {
        const double xi = std::sqrt(xi2);
        omega_ij = bij * std::tanh(0.5 * xi) / (2.0 * xi);
      }
      
      // Check if the Polya-Gamma Eq[z_ij] is doing fine
      if(!std::isfinite(omega_ij) || omega_ij < 0.0) Rcpp::stop("Invalid Polya-Gamma expectation encountered.");
      
      omega(i, j) = omega_ij;
    }
  }
}


// Function to update Sigma
void SpIOXNB::update_sigma_nbvi(){
  // same Sigma prior as Gaussian
  Sigma_scale = arma::eye<arma::mat>(core.q, core.q) + VTV_ma; 
  Sigma_scale = arma::symmatu(Sigma_scale);
  
  arma::mat inv_scale = arma::inv_sympd(Sigma_scale);
   
  // E_q[Sigma] = Sigma_scale / (df_post - q - 1)
  double df_post = core.n + core.q;
  Sigma_mean = Sigma_scale / (df_post - core.q - 1.0);
  
  // E_q[Sigma^{-1}] = df_post * Sigma_scale^{-1} 
  Qbar = arma::symmatu(df_post * inv_scale);
  //Qbar = arma::inv_sympd(Sigma_mean); 
  Qbar_sqrt = arma::chol(Qbar, "lower");
}


// The function to run a single iteration of nb latent VI
void SpIOXNB::nb_latent_vi(SpIOX::PrecondChoice preconditioner){
  // bookkeeping for runtime and stuff
  auto t0 = std::chrono::high_resolution_clock::now();
  auto t_prev = t0;
  auto checkpoint = [&](const char* label){
    auto t_now = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t_now - t_prev).count();
    // Rcpp::Rcout << "[latent_vi] " << label << ": " << ms << " ms\n";
    t_prev = t_now;
  };
  
  const arma::uvec missing = arma::find(core.missing_mat == 1);
  
  // update q(z)
  update_pg_expectation();
  checkpoint("Update q(z)");
  
  // build Gscore for the general_BW_block()
  arma::mat Gscore = kappa - omega % c;
  
  // assigning zeros for the missing obs (they don't contribute to the fitting)
  omega.elem(missing).zeros();
  Gscore.elem(missing).zeros();
  checkpoint("Update Gscore");
  
  // Draw from q(B, W)
  int cg_iter = 0;
  
  core.general_BW_block(omega, Gscore,
                        Qbar, Qbar_sqrt, cg_iter, preconditioner,
                        true,  // sampling mode
                        0,      // default PCG iteration cap
                        true);
  
  core.last_cg_iter = cg_iter;
  core.last_precond_used = static_cast<int>(preconditioner);
  checkpoint("Draw from BW_block");
  
  // Do the centering
  core.W_centering();  
  checkpoint("Centering #1");
  
  // Update psi and Eq[psi^2] with the MC sampled B W
  const arma::mat psi_draw = core.X * core.B + core.W + c; 
   
  core.update_running_means(Epsi2, arma::square(psi_draw)); 
  checkpoint("Compute and Smoothing Psi2");
  
  // Compute MC approximated V^T V and smoothing
  core.compute_V();
  const arma::mat VTV_draw = core.V.t() * core.V;
  core.update_running_means(VTV_ma, VTV_draw);
  checkpoint("Compute VTV and smoothing");
  
  // PX step and second W_centering and compute_V() should be done here!
  // Add after we have a NB compatible PX step
  
  // Update q(Sigma)
  update_sigma_nbvi();
  checkpoint("Update q(Sigma)");
  
  // Update UQ and posterior stuff for (B, W)
  core.vi_Beta_UQ();
  checkpoint("vi_Beta_UQ");
  
  core.update_running_means(core.E_B, core.B);
  core.update_running_means(core.E_W, core.W, false);
  checkpoint("update_running_means for B and W");
   
  double total_ms = std::chrono::duration<double, std::milli>(
    std::chrono::high_resolution_clock::now() - t0).count();
   
  core.vi_it++;
}

// evaluate the NB log-likelihood
double SpIOXNB::nb_latent_fit_eval() const{
  // for vi convergence
  const arma::mat eta = core.X * core.E_B + core.E_W + offset;
  
  double ll = 0.0;
  
  for(int j=0; j < core.q; j++){
    const double rj = r(j);
    const double log_rj = std::log(rj);
    
    // l_NB = \sum_{i=1}^n \sum_{j=1}^q [ log \Gamma(y_{ij}+r_j) - log \Gamma(r_j) 
    // - log \Gamma(y_{ij}+1) + r_j log r_j + y_{ij} \eta_{ij} - (y_{ij}+r_j) log{r_j+\exp(\eta_{ij}) }]
    // Compute for each (i, j) then sum
    for(int i = 0; i < core.n; i++){
      // skip the missing pairs
      if(core.missing_mat(i, j)) continue;
      
      const double yij = core.Y(i, j);
      const double etaij = eta(i, j);
      
      const double log_stuff = logaddexp_comp(log_rj, etaij);
      
      ll += std::lgamma(yij + rj) - std::lgamma(rj) - std::lgamma(yij + 1.0)
        + rj * log_rj + yij * etaij - (yij + rj) * log_stuff;
    } 
  }
  
  return ll;
}


// Function to do the latent VI predictive smapling for missing Yhat
// so we have a clear cut of the fitting part code and predicting part code
arma::vec SpIOXNB::nb_latent_vi_smp4Yhat() const{
  const arma::uword n_miss = core.Y_na_indices.n_elem;
  arma::vec out(n_miss, arma::fill::zeros);
  
  const double prob_floor = std::numeric_limits<double>::min();
  const double prob_ceil = 1.0 - std::numeric_limits<double>::epsilon();
  
  // For each missing (i, j), compute the prob_ij = r_j / (r_j + exp(X_i^T B_j + W_ij + offset_ij))
  // Then sample Y_ij | r_j, prob_ij ~ NB(r_j, prob_ij) 
  for(arma::uword k = 0; k < n_miss; ++k){
    const arma::uword idx = core.Y_na_indices(k);
    const arma::uword i = idx % core.n;
    const arma::uword j = idx / core.n;
    
    const double eta_ij = arma::as_scalar(core.X.row(i) * core.B.col(j)) + core.W(i, j) + offset(i, j);
      
      // logistic_comp input log (prob_ij) and returns prob_ij
      double prob_ij = logistic_comp(std::log(r(j)) - eta_ij);
      
      // keeping probability in range
      prob_ij = std::min(prob_ceil, std::max(prob_floor, prob_ij));
      
      out(k) = R::rnbinom(r(j), prob_ij);
  }
  
  return out;
}
