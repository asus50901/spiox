#include "spiox.h"
#include "interrupt.h"
#include <RcppArmadillo.h>
#include <chrono>
using namespace Rcpp;

//[[Rcpp::export]]
arma::field<arma::sp_mat> spiox_H_list(const arma::mat& coords,
                                       const arma::field<arma::uvec>& custom_dag,
                                       const arma::mat& Theta,
                                       int covariance_matern = 1,
                                       int num_threads = 1){
  
  int q = Theta.n_cols;
  arma::field<arma::sp_mat> Hlist(q);
  for(int j=0; j<q; j++){
    DagGP daggp(coords, Theta.col(j), custom_dag, 0, covariance_matern, false, num_threads);
    Hlist(j) = daggp.H;
  }
  return Hlist;
}

//[[Rcpp::export]]
arma::mat spiox_simulate(const arma::mat& coords,
                          const arma::field<arma::uvec>& custom_dag,
                          const arma::mat& Sigma,
                          const arma::mat& Theta,
                          int covariance_matern = 1,
                          int num_threads = 1){
  
  unsigned int n = coords.n_rows;
  unsigned int q = Theta.n_cols;
  
  // generate white noise and correlate across margins
  arma::mat St = arma::chol(Sigma, "upper");
  arma::mat U = arma::randn(n, q);
  arma::mat V = U * St;
  
  arma::mat W = arma::zeros(n, q);
  
  // induce spatial dependence
  for(unsigned int j=0; j<q; j++){
    DagGP daggp(coords, Theta.col(j), custom_dag, 0, covariance_matern, false, num_threads);
    
    W.col(j) = daggp.H_solve_A(V.col(j));
    W.col(j) -= arma::mean(W.col(j));
  }
  
  return(W);
}

// [[Rcpp::export]]
Rcpp::List spiox_response(const arma::mat& Y, 
                    const arma::mat& X, 
                    const arma::mat& coords,
                    
                    const arma::field<arma::uvec>& custom_dag,
                
                    const arma::mat& Beta_start,
                    const arma::mat& Sigma_start,
                    const arma::mat& Theta_start, 
                    
                    int mcmc = 1000,
                    int print_every = 100,
                    int matern = 1,
                    int dag_opts = 0,
                    bool sample_Beta = true,
                    bool sample_Sigma = true,
                    const arma::uvec& update_Theta = arma::ones<arma::uvec>(4),
                    int num_threads = 1){
  
#ifdef _OPENMP
  omp_set_num_threads(num_threads);
#else
  if(num_threads > 1){
    Rcpp::warning("num_threads > 1, but source not compiled with OpenMP support.");
    num_threads = 1;
  }
#endif
  
  int latent_model = 0;
  
  unsigned int q = Y.n_cols;
  unsigned int n = Y.n_rows;
  
  int sample_precision = 2 * sample_Sigma;
  
  std::string fit_descr = "with posterior sampling of Theta";
  if(arma::all(update_Theta == 0)){
    fit_descr = "with fixed Theta";
  } 

  if(print_every > 0){
    Rcpp::Rcout << "Preparing for GP-IOX response model, MCMC " << fit_descr << endl;
  }
  
  
  // tausq not needed in this model
  arma::vec tausq_not_needed = arma::zeros(q);
  arma::mat W_not_needed = arma::zeros(1,1);
  
  SpIOX iox_model(Y, X, coords, custom_dag, dag_opts,
                  latent_model,
                  Beta_start,
                  W_not_needed, // not used
                   Sigma_start,
                   Theta_start, 
                   update_Theta,
                   tausq_not_needed,
                   matern,
                   num_threads, 0); // 0 for vi_min_iter
  
  int nmiss = 0;
  if(iox_model.Y_needs_filling){
    nmiss = iox_model.Y_na_indices.n_elem;
  }
  
  // storage
  arma::cube Beta = arma::zeros(iox_model.p, q, mcmc);
  arma::cube Sigma = arma::zeros(q, q, mcmc);
  arma::cube theta = arma::zeros(4, q, mcmc);
  
  // only store samples of imputed Y if missing
  arma::mat Y_missing_fill = arma::zeros(nmiss, mcmc);
  
  if(print_every > 0){
    Rcpp::Rcout << "Starting MCMC" << endl;
  }
  bool theta_needs_updating = arma::any(update_Theta == 1);
  
  for(unsigned int m=0; m<mcmc; m++){
    
    iox_model.response_gibbs(m, sample_precision, sample_Beta, theta_needs_updating);

    Beta.slice(m) = iox_model.B;
    Sigma.slice(m) = iox_model.Sigma;
    theta.slice(m) = iox_model.theta;
    if(iox_model.Y_needs_filling){
      Y_missing_fill.col(m) = iox_model.Y(iox_model.Y_na_indices);
    }
    
    bool print_condition = (print_every>0);
    if(print_condition){
      print_condition = print_condition & (!(m % print_every));
    };
    if(print_condition){
      Rcpp::Rcout << "Iteration: " <<  m+1 << " of " << mcmc << endl;
    }
    
    bool interrupted = checkInterrupt();
    if(interrupted){
      Rcpp::stop("Interrupted by the user.");
    }
  }

  return Rcpp::List::create(
    Rcpp::Named("Beta") = Beta,
    Rcpp::Named("Sigma") = Sigma,
    Rcpp::Named("Theta") = theta,
    Rcpp::Named("Y_missing_samples") = Y_missing_fill,
    Rcpp::Named("Y_missing_indices") = iox_model.Y_na_indices+1,
    Rcpp::Named("timings") = iox_model.timings,
    Rcpp::Named("dag_cache") = iox_model.daggps[0].dag_cache
  );
  
}




// [[Rcpp::export]]
Rcpp::List spiox_latent(const arma::mat& Y, 
                          const arma::mat& X, 
                          const arma::mat& coords,
                          
                          const arma::field<arma::uvec>& custom_dag,
                          
                          const arma::mat& Beta_start,
                          const arma::mat& W_start,
                          const arma::mat& Sigma_start,
                          const arma::mat& Theta_start, 
                          const arma::vec& Ddiag_start,
                          
                          int mcmc=1000,
                          int print_every=100,
                          int matern = 1,
                          int dag_opts = 0,
                          bool sample_Beta=true,
                          bool sample_Sigma=true,
                          bool sample_Ddiag=true,
                          const arma::uvec& update_Theta = arma::ones<arma::uvec>(4),
                          int num_threads = 1, 
                          int sampling=2){
  
  
  if(sampling==0){
    Rcpp::stop("Invalid MCMC sampling option. Choose 1/2/3.");
  }
  
  std::string fit_descr = "single-site sampler (n sequential, q block)";
  if(sampling==1){
    fit_descr = "block sampler (nq block) ~ slow!";
  }
  if(sampling==3){
    fit_descr = "single-outcome sampler (q sequential, n block)";
  }
  
#ifdef _OPENMP
  omp_set_num_threads(num_threads);
#else
  if(num_threads > 1){
    Rcpp::warning("num_threads > 1, but source not compiled with OpenMP support.");
    num_threads = 1;
  }
#endif
  
  unsigned int q = Y.n_cols;
  unsigned int n = Y.n_rows;
  
  int sample_precision = 2 * sample_Sigma;
  
  
  if(print_every > 0){
    Rcpp::Rcout << "Preparing for GP-IOX latent model, MCMC " << fit_descr << endl;
  }
  
  SpIOX iox_model(Y, X, coords, custom_dag, dag_opts,
                  sampling,
                  Beta_start,
                  W_start,
                  Sigma_start,
                  Theta_start, 
                  update_Theta,
                  Ddiag_start,
                  matern,
                  num_threads, 0); // 0 for vi_min_iter
  
  // storage
  arma::cube Beta = arma::zeros(iox_model.p, q, mcmc);
  arma::cube Sigma = arma::zeros(q, q, mcmc);
  arma::cube theta = arma::zeros(4, q, mcmc);
  arma::mat Ddiag = arma::zeros(q, mcmc);
  arma::cube W = arma::zeros(n, q, mcmc);
  
  if(print_every > 0){
    Rcpp::Rcout << "Starting MCMC" << endl;
  }
  
  bool theta_needs_updating = arma::any(update_Theta == 1);
  
  for(unsigned int m=0; m<mcmc; m++){
    
    iox_model.latent_gibbs(m, sample_precision, sample_Beta, theta_needs_updating, sample_Ddiag);
    
    Beta.slice(m) = iox_model.B;
    Sigma.slice(m) = iox_model.Sigma;
    theta.slice(m) = iox_model.theta;
    Ddiag.col(m) = iox_model.Ddiag;
    W.slice(m) = iox_model.W;

    bool print_condition = (print_every>0);
    if(print_condition){
      print_condition = print_condition & (!(m % print_every));
    };
    if(print_condition){
      Rcpp::Rcout << "Iteration: " <<  m+1 << " of " << mcmc << endl;
    }
    
    bool interrupted = checkInterrupt();
    if(interrupted){
      Rcpp::stop("Interrupted by the user.");
    }
  }
  
  return Rcpp::List::create(
    Rcpp::Named("Beta") = Beta,
    Rcpp::Named("Sigma") = Sigma,
    Rcpp::Named("Theta") = theta,
    Rcpp::Named("W") = W,
    Rcpp::Named("Ddiag") = Ddiag,
    Rcpp::Named("timings") = iox_model.timings,
    Rcpp::Named("markov_blanket") = iox_model.daggps[0].mblanket
  );
}


// [[Rcpp::export]]
Rcpp::List spiox_response_vi(const arma::mat& Y, 
                          const arma::mat& X, 
                          const arma::mat& coords,
                          
                          const arma::field<arma::uvec>& custom_dag,
                          int dag_opts,
                          const arma::mat& Theta, 
                          
                          const arma::mat& Sigma_start,
                          const arma::mat& Beta_start,
                          
                          int print_every = 0,
                          int matern = 1,
                          int num_threads = 1){
  
  double tol = 1e-5;
  int min_iter = 1;
  int max_iter = 500;
  
  // missing data not allowed here
  arma::uvec y_missing_ix = arma::find_nonfinite(Y);
  if(y_missing_ix.n_elem > 0){
    Rcpp::stop("Missing values in Y not allowed in response-vi. Choose response-mcmc or a latent model.");
  }
  
#ifdef _OPENMP
  omp_set_num_threads(num_threads);
#else
  if(num_threads > 1){
    Rcpp::warning("num_threads > 1, but source not compiled with OpenMP support.");
    num_threads = 1;
  }
#endif
  
  int latent_model = 0;
  
  unsigned int p = X.n_cols;
  unsigned int q = Y.n_cols;
  unsigned int n = Y.n_rows;
  
  if(print_every > 0){
    Rcpp::Rcout << "Preparing for GP-IOX response model, VI fit..." << endl;
  }
  
  // tausq not needed in this model
  arma::vec tausq_not_needed = arma::zeros(q);
  arma::uvec not_updating_theta = arma::zeros<arma::uvec>(4);
  arma::mat W_not_needed = arma::zeros(1,1);
  
  SpIOX iox_model(Y, X, coords, custom_dag, dag_opts,
                  latent_model,
                  
                  Beta_start,
                  W_not_needed, 
                  Sigma_start,
                  Theta, 
                  not_updating_theta,
                  tausq_not_needed,
                  matern,
                  num_threads, min_iter);
  
  // storage
  arma::mat Beta = arma::zeros(p, q);
  arma::mat Sigma = arma::zeros(q, q);
  
  if(print_every > 0){
    Rcpp::Rcout << "Starting VI." << endl;
  }
  
  bool stop=false;
  int i=0;
  while(!stop){
    i ++;
    
    arma::mat Beta_pre = Beta;
    arma::mat Sigma_pre = Sigma;
    
    iox_model.response_vi();
    
    Beta = iox_model.B;
    Sigma = iox_model.Sigma;
    
    double rel_mu_change = arma::norm(Beta - Beta_pre, 2) / (arma::norm(Beta_pre, 2) + 1e-12);
    double rel_sigma_change = arma::norm(Sigma - Sigma_pre, "fro") / (arma::norm(Sigma_pre, "fro") + 1e-12);
    
    if (rel_mu_change < tol && rel_sigma_change < tol && i > min_iter) {
      stop=true;
    }
    
    bool print_condition = (print_every>0);
    if(print_condition){
      print_condition = print_condition & (!(i % print_every));
    };
    if(print_condition){
      Rcpp::Rcout << "Iteration: " <<  i << endl;
    }
    
    bool interrupted = checkInterrupt();
    if(interrupted){
      Rcpp::stop("Interrupted by the user.");
    }
  }
  
  // mfvi
  arma::mat Beta_UQ = iox_model.Beta_UQ;
  
  //arma::vec vecBeta_UQ = iox_model.Beta_UQ.diag();
  //arma::mat Beta_UQ = arma::mat(vecBeta_UQ.memptr(), p, q);
  
  return Rcpp::List::create(
    Rcpp::Named("Beta") = Beta,
    Rcpp::Named("Beta_UQ") = Beta_UQ,
    Rcpp::Named("Sigma") = Sigma,
    Rcpp::Named("Sigma_UQ") = iox_model.Sigma_UQ
  );
  
}



// [[Rcpp::export]]
Rcpp::List spiox_latent_vi(const arma::mat& Y, 
                           const arma::mat& X, 
                           const arma::mat& coords,
                           
                           const arma::field<arma::uvec>& custom_dag,
                           int dag_opts,
                           const arma::mat& Theta, 
                           
                           const arma::mat& Sigma_start,
                           const arma::mat& Beta_start,
                           const arma::mat& W_start,
                           const arma::vec& Ddiag_start,
                           
                           int matern = 1,
                           int num_threads = 1,
                           int print_every = 0,
                           double tol = 1e-2,
                           int max_iter = 500,
                           int vi_pred_smp = 0){
  
  // do min_iter iterations at least
  int nq = Y.n_cols * Y.n_rows;
  int min_iter = 200; //min(50, nq / 1000); 
  // then check maximum relative change. if it's <tol for this time then stop
  int wait_time_before_stop = 5;
  
  // for artifacts in other subfunctions.
  int latent_model = 2; 
  
  
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
  
  if(print_every > 0){
    Rcpp::Rcout << "Preparing for GP-IOX latent model, VI fit..." << endl;
  }
  
  
  arma::uvec not_updating_theta = arma::zeros<arma::uvec>(4); //
  
  SpIOX iox_model(Y, X, coords, custom_dag, dag_opts,
                  latent_model,
                  
                  Beta_start,
                  W_start,
                  Sigma_start,
                  Theta, 
                  not_updating_theta,
                  Ddiag_start, // Need tau_sq for latent VI 
                  matern,
                  num_threads, min_iter);
  
  double ll_pre = iox_model.latent_fit_eval();
  
  // storage
  arma::mat Beta = arma::zeros(p, q);
  arma::mat Sigma = arma::zeros(q, q);
  arma::mat W = arma::zeros(n, q);
  arma::vec Ddiag = arma::zeros(q);
  
  // for trace plots
  arma::vec rel_B_store(max_iter, arma::fill::zeros);
  arma::vec rel_Sigma_store(max_iter, arma::fill::zeros);
  arma::vec rel_W_store(max_iter, arma::fill::zeros);
  arma::vec rel_D_store(max_iter, arma::fill::zeros);
  arma::vec rel_ll_store(max_iter, arma::fill::zeros);
  
  // for misalignment prediction
  int n_miss = 0;
  if(iox_model.Y_needs_filling){
    n_miss = iox_model.Y_na_indices.n_elem;
  }
  
  arma::cube Beta_post_samples;
  arma::cube W_post_samples;
  arma::mat Ddiag_post_samples;
  arma::mat Y_missing_samples;
  
  if(vi_pred_smp < 0){
    Rcpp::stop("vi_pred_smp must be >= 0.");
  } 
  
  if(vi_pred_smp > 0){
    Beta_post_samples = arma::zeros(p, q, vi_pred_smp);
    W_post_samples = arma::zeros(n, q, vi_pred_smp);
    Ddiag_post_samples = arma::zeros(q, vi_pred_smp);
    Y_missing_samples = arma::zeros(n_miss, vi_pred_smp);
  } 
  
  if(print_every > 0){
    Rcpp::Rcout << "Starting VI." << endl;
  }
  
  bool stop=false; // stopping flag
  int about_to_exit = 0; // counter for how long we've been "good"
  
  // additional flags for collecting misalignment pred smp
  bool converged = false;
  bool collecting = false;
  int collect_counter = 0;
  
  int i=0;
  while(!stop){
    i ++;
    
    arma::mat Beta_pre = Beta;
    arma::mat Sigma_pre = Sigma;
    arma::mat W_pre = iox_model.E_W; 
    arma::vec D_pre = iox_model.Ddiag;
    
    
    iox_model.latent_vi();
    
    Beta = iox_model.E_B;
    Sigma = iox_model.Sigma;
    W = iox_model.E_W;
    Ddiag = iox_model.Ddiag;
    double ll = iox_model.latent_fit_eval();
    
    // monitoring convergence of the parameters
    arma::vec rel_change = arma::zeros(5);
    rel_change(0) = arma::norm(Beta - Beta_pre, "fro") / (arma::norm(Beta_pre, "fro")  + 1e-12);
    rel_change(1) = arma::norm(Sigma - Sigma_pre, "fro") / (arma::norm(Sigma_pre, "fro") + 1e-12);
    rel_change(2) = arma::norm(W - W_pre, "fro") / (arma::norm(W_pre, "fro") + 1e-12);
    rel_change(3) = arma::norm(Ddiag - D_pre, 2) / (arma::norm(D_pre, 2) + 1e-12);
    rel_change(4) = abs(ll - ll_pre) / (abs(ll_pre) + 1e-12);
    
    ll_pre = ll;
    
    double max_rel_change = rel_change.max();
    
    bool print_condition = (print_every>0);
    if(print_every > 0 && !collecting){
      if(i % print_every == 0){
        Rcpp::Rcout << "Iteration: " << i << endl;
      }
    }
    
    // storing for optimizing trace plots
    if(!collecting && i <= max_iter){
      rel_B_store(i - 1) = rel_change(0);
      rel_Sigma_store(i - 1) = rel_change(1);
      rel_W_store(i - 1) = rel_change(2);
      rel_D_store(i - 1) = rel_change(3);
      rel_ll_store(i - 1) = rel_change(4);
    }
    
    if(i > min_iter){
      if(max_rel_change < tol){
        // we're doing well, prepare to exit
        about_to_exit += 1;
      } else {
        // reset
        about_to_exit = 0;
      }
    }
    
    bool interrupted = checkInterrupt();
    if(interrupted){
      Rcpp::stop("Interrupted by the user.");
    }
    
    // stopping?
    //stop = about_to_exit > wait_time_before_stop;
    //if (i >= max_iter) stop = true;
    bool stop_trigger = about_to_exit > wait_time_before_stop;
    if(!collecting && stop_trigger){
      converged = true;
      
      if(vi_pred_smp == 0){
        stop = true;
      } else{
        collecting = true; 
        collect_counter = 0;
        if(print_every > 0){
          Rcpp::Rcout << "Starting VI missing sample collection" << endl;
        }
      }
    } 
    
    // max_iter is only for the optimization, not for pred samples
    if(!collecting && i >= max_iter){
      stop = true;
    } 
    
    if(collecting){
      if(collect_counter < vi_pred_smp){
        
        // smp for misalign pred
        arma::mat Beta_draw = iox_model.B;
        arma::mat W_draw = iox_model.W;
        
        // Draw Ddiag from the inverse Wishart
        arma::vec Ddiag_draw(q);
        for(unsigned int j = 0; j < q; j++){
          double shape = (iox_model.avail_by_outcome(j).n_elem - 1.0) / 2.0 + 2.0;
          double rate  = 1.0 + 0.5 * iox_model.Ddiag_UQ(j);
          Ddiag_draw(j) = 1.0 / R::rgamma(shape, 1.0 / rate);
        } 
        
        // storing 
        Beta_post_samples.slice(collect_counter) = Beta_draw;
        W_post_samples.slice(collect_counter) = W_draw;
        Ddiag_post_samples.col(collect_counter) = Ddiag_draw;
        
        // Generate the Noise term
        arma::mat noise = arma::randn(n, q);
        noise.each_row() %= arma::sqrt(Ddiag_draw).t();
        
        arma::mat Yhat = X * Beta_draw + W_draw + noise;
        
        if(n_miss > 0){
          Y_missing_samples.col(collect_counter) = Yhat(iox_model.Y_na_indices);
        } 
        
        collect_counter++;
        if(print_every > 0){
          if(collect_counter % print_every == 0){
            Rcpp::Rcout << "VI missing sample: " << collect_counter
                        << " of " << vi_pred_smp << endl;
          }
        } 
      }
      
      if(collect_counter >= vi_pred_smp){
        stop = true;
      } 
    }
    
  } 
  
  //arma::vec vecBeta_UQ = iox_model.Beta_UQ.diag() / (i-1.0);
  //arma::mat Beta_UQ = arma::mat(vecBeta_UQ.memptr(), p, q);
  
  // cut off the unused(NA) elements
  int n_trace = std::min(i - collect_counter, max_iter);
  rel_B_store = rel_B_store.head(n_trace);
  rel_Sigma_store = rel_Sigma_store.head(n_trace);
  rel_W_store = rel_W_store.head(n_trace);
  rel_D_store = rel_D_store.head(n_trace);
  rel_ll_store = rel_ll_store.head(n_trace);
  
  arma::mat Beta_UQ = iox_model.Beta_UQ/(i-1.0-min_iter);
  
  return Rcpp::List::create(
    Rcpp::Named("Beta") = Beta,
    Rcpp::Named("Beta_UQ") = Beta_UQ,
    Rcpp::Named("Sigma") = Sigma,
    Rcpp::Named("Sigma_UQ") = iox_model.Sigma_UQ,
    Rcpp::Named("W") = W,
    Rcpp::Named("Ddiag") = Ddiag,
    Rcpp::Named("Ddiag_UQ") = iox_model.Ddiag_UQ,
    Rcpp::Named("rel_change") = Rcpp::List::create(
      Rcpp::Named("rel_Beta")  = rel_B_store,
      Rcpp::Named("rel_Sigma") = rel_Sigma_store,
      Rcpp::Named("rel_W")     = rel_W_store,
      Rcpp::Named("rel_Ddiag") = rel_D_store,
      Rcpp::Named("rel_ll") = rel_ll_store
    ),
    Rcpp::Named("n_iter") = converged ?(i - collect_counter):i,
    Rcpp::Named("Y_missing_samples") = Y_missing_samples,
    Rcpp::Named("Y_missing_indices") = iox_model.Y_na_indices + 1,
    Rcpp::Named("markov_blanket") = iox_model.daggps[0].mblanket
  );
  
}
