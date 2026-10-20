#include "spiox.h"

//[[Rcpp::export]]
Rcpp::List spiox_denoiseY(
    const arma::mat& X_new,
    const arma::mat& coords_new,
    
    const arma::mat& Y, 
    const arma::mat& X, 
    const arma::mat& coords,
    
    const arma::field<arma::uvec>& dag,
    
    const arma::cube& B,
    const arma::cube& theta,
    int matern = 1,
    int num_threads = 1,
    int print_every = 100
){
  int q = Y.n_cols;
  int p = X.n_cols;
  int mcmc = B.n_slices;
  
  arma::uvec oneuv = arma::ones<arma::uvec>(1);
  
  int bessel_ws_inc = MAT_NU_MAX;//see bessel_k.c for working space needs
  double * bessel_ws = (double *) R_alloc(num_threads*bessel_ws_inc, sizeof(double));
  
  int ntrain = coords.n_rows;
  int ntest = coords_new.n_rows;
  
  arma::mat cxall = arma::join_vert(coords, coords_new);
  
  // an alert to remind myself of sigmasq = 1
  if(theta.n_rows != 4) Rcpp::stop("theta must have 4 rows: phi, sigmasq, nu, alpha.");
  
  //storage
  arma::cube denoise_Y_mcmc = arma::zeros(ntest, q, mcmc);
  arma::cube pseudo_W_mcmc = arma::zeros(ntest, q, mcmc);
  
  //compute posterior draws before looping over locations
  for(int m=0; m<mcmc; m++){
    arma::mat B_m = B.slice(m);
    arma::mat theta_m = theta.slice(m);
    
    //compute once then reuse 
    arma::mat XB_old = X * B_m;
    arma::mat YXB = Y - XB_old;
    
    arma::mat XB_new = X_new * B_m;
    
    //loop over test locations
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
    for(int i=0; i<ntest; i++){
      
      arma::uvec ix = oneuv * i + ntrain;
      arma::uvec px = dag(i);
      
      arma::vec Yspat = arma::zeros(q);
      arma::vec Y_out = arma::zeros(q);
      
      //loop over outcomes
      for(int j=0; j<q; j++){
        // predict spatially
        arma::uvec outjx = oneuv * j;
        
        arma::mat CPt = Correlationf(cxall, px, ix, theta_m.col(j), bessel_ws, matern, false);
        arma::mat CPP = Correlationf(cxall, px, px, theta_m.col(j), bessel_ws, matern, true);
        
        arma::mat ht = arma::solve(CPP, CPt, arma::solve_opts::likely_sympd);
        
        Yspat(j) = arma::conv_to<double>::from(ht.t() * YXB(px, outjx));
        Y_out(j) = XB_new(i, j) + Yspat(j); // post mean only
      }
      
      for(int j=0; j<q; j++){
        denoise_Y_mcmc(i, j, m) = Y_out(j);
        pseudo_W_mcmc(i, j, m) = Yspat(j);
      }
    }
    
    //progress reporting
    if(print_every > 0 && (m + 1) % print_every == 0){
      Rcpp::Rcout << "Completed " << (m + 1) << " / " << mcmc << " draws ("
                  << 100.0 * (m + 1) / mcmc << "%)" << std::endl;
    }
    //Rcpp::Rcout << "saving m" << endl;
    //denoise_Y_mcmc.slice(m) = Y_out;
  }
  
  return Rcpp::List::create(
    Rcpp::Named("Y_denoised") = denoise_Y_mcmc,
    Rcpp::Named("W_pseudo") = pseudo_W_mcmc,
    Rcpp::Named("dag") = dag 
  );
}
