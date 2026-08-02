#include "spiox_nb.h"
#include "interrupt.h"
#include <RcppArmadillo.h> 
#include <chrono>
#include <sstream>
#include <iomanip>

using namespace Rcpp;

// Where we put the Rcpp export function that builds the NB model, run tierations, 
// monitor convergence, predict, and return R list

// Progress suffix for the VI fit.  Like mcmc_eta_str but WITHOUT an ETA: VI runs
// to convergence, so the total iteration count is not known in advance.
static std::string nb_vi_progress_str(std::chrono::steady_clock::time_point t0,
                               std::chrono::steady_clock::time_point& last_print){
  auto now = std::chrono::steady_clock::now();
  double elapsed_s    = std::chrono::duration<double>(now - t0).count();
  double since_last_s = std::chrono::duration<double>(now - last_print).count();
  last_print = now;
  std::ostringstream os;
  os << std::fixed << std::setprecision(1)
     << "[elapsed " << elapsed_s << "s | +" << since_last_s << "s]";
  return os.str();
}


// [[Rcpp::export]]
Rcpp::List spiox_nb_latent_vi(const arma::mat& Y, 
                              const arma::mat& X, 
                              const arma::mat& coords,
                              const arma::mat& offset, 
                              const arma::vec& r,
                              
                              const arma::field<arma::uvec>& custom_dag,
                              int dag_opts,
                              const arma::mat& Theta, 
                              
                              const arma::mat& Sigma_start,
                              const arma::mat& Beta_start,
                              const arma::mat& W_start,
                              
                              int matern = 1,
                              int num_threads = 1,
                              int print_every = 0,
                              double tol = 1e-2,
                              int max_iter = 500,
                              int vi_pred_smp = 0,
                              int cg_preconditioner = 0){
  
  // cg_preconditioner selector for the VI block (B,W) CG solve:
  //   0 = auto    (resolves to POSTCOV for the latent VI fit, see latent_vi())
  //   1 = JACOBI  2 = VADU  3 = POSTCOV
  // VI always uses the joint block sampler, so every choice is honoured directly
  // (no sampling=3 per-outcome fallback as in MCMC).
  if(cg_preconditioner < 0 || cg_preconditioner > 3){
    Rcpp::stop("cg_preconditioner must be in {0,1,2,3} (auto/jacobi/vadu/postcov).");
  }
  
  // do min_iter iterations at least
  int nq = Y.n_cols * Y.n_rows;
  int min_iter = 40;
  // then check maximum relative change. if it's <tol for this time then stop
  int wait_time_before_stop = 5;
  
  // for artifacts in other subfunctions.
  int latent_model = 1; 
  
#ifdef _OPENMP
  omp_set_num_threads(num_threads);
#else
  if(num_threads > 1){
    Rcpp::warning("num_threads > 1, but source not compiled with OpenMP support.");
    num_threads = 1;
  }
#endif
  
  unsigned int p = X.n_cols;
  unsigned int q = Y.n_cols;
  unsigned int n = Y.n_rows;
  
  const SpIOX::PrecondChoice preconditioner =
    cg_preconditioner == 0 ? SpIOX::PRECOND_VADU : static_cast<SpIOX::PrecondChoice>(cg_preconditioner);
  
  if(print_every > 0){
    Rcpp::Rcout << "Preparing for negative binomial GP-IOX latent model, VI fit..." << endl;
  }
  
  if(vi_pred_smp < 0){
    Rcpp::stop("vi_pred_smp must be >= 0.");
  } 
  
  // something for MCMC only.
  //arma::uvec not_updating_theta = arma::zeros<arma::uvec>(4); 
  
  SpIOXNB nb_model(Y, X, coords, custom_dag, dag_opts,
                   
                   Theta, 
                   Beta_start,
                   W_start,
                   Sigma_start,
                   r, offset,
                   
                   matern,
                   num_threads, min_iter);
  
  double ll_pre = nb_model.nb_latent_fit_eval();
  
  // constructor for class SpIOXNB already initialize Epsi2 and omega (so fine after nb_model(...))
  // all the B, W, Sigma initializing is also handled in the constructor
  
  // for trace plots
  arma::vec rel_B_store(max_iter, arma::fill::zeros);
  arma::vec rel_Sigma_store(max_iter, arma::fill::zeros);
  arma::vec rel_W_store(max_iter, arma::fill::zeros);
  arma::vec rel_omega_store(max_iter, arma::fill::zeros);
  arma::vec rel_ll_store(max_iter, arma::fill::zeros);
  
  if(print_every > 0){
    Rcpp::Rcout << "Starting negative binomial Polya-Gamma VI." << endl;
  }
   
  // flags for stopping and stuff
  bool converged = false; 
  int exit_counter = 0;
  int iter_done = 0;
  
  auto vi_t0 = std::chrono::steady_clock::now();
  auto vi_last_print = vi_t0;
  
  for(int iter = 0; iter<max_iter; ++iter){
    
    // update the previous state for params
    arma::mat Beta_pre = nb_model.core.E_B;
    arma::mat Sigma_pre = nb_model.Sigma_mean;
    arma::mat W_pre = nb_model.core.E_W; 
    arma::mat omega_pre = nb_model.omega;
    
    // run a single VI iteration
    nb_model.nb_latent_vi(preconditioner);
    double ll = nb_model.nb_latent_fit_eval();
    
    iter_done ++;
    
    // monitoring convergence of the parameters
    arma::vec rel_change(5, arma::fill::zeros);
    rel_change(0) = arma::norm(nb_model.core.E_B - Beta_pre, "fro") / (arma::norm(Beta_pre, "fro") + 1e-12);
    rel_change(1) = arma::norm(nb_model.Sigma_mean - Sigma_pre, "fro") / (arma::norm(Sigma_pre, "fro") + 1e-12);
    rel_change(2) = arma::norm(nb_model.core.E_W - W_pre, "fro") / (arma::norm(W_pre, "fro") + 1e-12);
    rel_change(3) = arma::norm(nb_model.omega - omega_pre, "fro") / (arma::norm(omega_pre, "fro") + 1e-12);
    rel_change(4) = std::abs(ll - ll_pre) / (std::abs(ll_pre) + 1e-12);
    
    ll_pre = ll;
    double max_rel_change = rel_change.max();
    
    // store the relative changes for trace plot
    rel_B_store(iter) = rel_change(0);
    rel_Sigma_store(iter) = rel_change(1);
    rel_W_store(iter) = rel_change(2);
    rel_omega_store(iter) = rel_change(3);
    rel_ll_store(iter) = rel_change(4);
    
    if(print_every > 0 && (iter_done % print_every == 0)){
      const char* pc_name =
        nb_model.core.last_precond_used == 1 ? "jacobi"  :
      nb_model.core.last_precond_used == 2 ? "vadu"    :
      nb_model.core.last_precond_used == 3 ? "postcov" : "n/a";
      std::ostringstream rc;
      rc << std::scientific << std::setprecision(2) << max_rel_change;
      Rcpp::Rcout << "Iteration: " << iter_done
                  << "  (CG: " << nb_model.core.last_cg_iter
                  << " iters, pc=" << pc_name << ")  "
                  << "max rel change " << rc.str() << "  "
                  << nb_vi_progress_str(vi_t0, vi_last_print) << std::endl;
    }
    
    // stop the fitting?
    if(iter_done > min_iter){
      if(max_rel_change < tol){
        // we're doing well, prepare to exit
        ++exit_counter;
      } else {
        // reset
        exit_counter = 0;
      }
    }
    
    if(exit_counter > wait_time_before_stop){
      converged = true;
      break;
    }
    
    if(checkInterrupt()){
      Rcpp::stop("Interrupted by the user.");
    }
    
  } // end of model fitting for loop
  
  // cut off the unused(NA) elements for the trace 
  const arma::uword n_trace = static_cast<arma::uword>(iter_done);
  
  rel_B_store = rel_B_store.head(n_trace);
  rel_Sigma_store = rel_Sigma_store.head(n_trace);
  rel_W_store = rel_W_store.head(n_trace);
  rel_omega_store = rel_omega_store.head(n_trace);
  rel_ll_store = rel_ll_store.head(n_trace);
  
  // Prep for VI missing sample Yhat collection
  int n_miss = 0;
  if(nb_model.core.Y_needs_filling){
    n_miss = nb_model.core.Y_na_indices.n_elem;
  }
  
  // misalignment prediction storage
  arma::mat Y_missing_samples(n_miss, 0);
  int n_pred_collected = 0;
  
  if(vi_pred_smp > 0 && converged){ 
    Y_missing_samples.set_size(n_miss, vi_pred_smp);
    Y_missing_samples.zeros();
    
    if(print_every > 0){
      Rcpp::Rcout << "Starting VI missing sample collection" << endl;
    }
    
    for(int s = 0; s < vi_pred_smp; ++s){
      // continue the VI algorithm to collect Yhat
      // That's fine bcs we converged.
      nb_model.nb_latent_vi(preconditioner);
      
      // predict for Yhat
      if(n_miss > 0){
        Y_missing_samples.col(s) = nb_model.nb_latent_vi_smp4Yhat();
      }
      
      ++n_pred_collected;
      
      if(print_every > 0 && n_pred_collected % print_every == 0){
        Rcpp::Rcout << "VI missing sample: " << n_pred_collected
        << " of " << vi_pred_smp << std::endl;
      }
      
      if(checkInterrupt()){
        Rcpp::stop("Interrupted by the user.");
      }
      
    } // end of misalignment prediction for loop
    
  } else if(vi_pred_smp > 0 && !converged){
    Rcpp::warning("VI reached max_iter before converging; predictive samples not collected.");
  }
  
  
  //arma::vec vecBeta_UQ = iox_model.Beta_UQ.diag() / (i-1.0);
  //arma::mat Beta_UQ = arma::mat(vecBeta_UQ.memptr(), p, q);
  double denom_betaUQ = std::max(1.0, (nb_model.core.vi_it - nb_model.core.vi_min_iter - 1.0));
  
  arma::mat Beta_UQ = nb_model.core.Beta_UQ / denom_betaUQ;
  
  return Rcpp::List::create(
    Rcpp::Named("Beta") = nb_model.core.E_B,
    Rcpp::Named("Beta_UQ") = Beta_UQ,
    Rcpp::Named("Sigma") = nb_model.Sigma_mean,
    Rcpp::Named("Sigma_UQ") = nb_model.Sigma_scale,
    Rcpp::Named("W") = nb_model.core.E_W, 
    Rcpp::Named("Theta") = Theta,
    Rcpp::Named("r") = r,
    Rcpp::Named("offset") = offset,
    Rcpp::Named("rel_change") = Rcpp::List::create(
      Rcpp::Named("rel_Beta")  = rel_B_store,
      Rcpp::Named("rel_Sigma") = rel_Sigma_store,
      Rcpp::Named("rel_W")     = rel_W_store,
      Rcpp::Named("rel_omega") = rel_omega_store,
      Rcpp::Named("rel_ll") = rel_ll_store
    ),
    Rcpp::Named("n_iter") = iter_done,
    Rcpp::Named("Y_missing_samples") = Y_missing_samples,
    Rcpp::Named("Y_missing_indices") = nb_model.core.Y_na_indices + 1,
    Rcpp::Named("markov_blanket") = nb_model.core.daggps[0].mblanket
  );
  
}