#include "spiox.h"

using namespace std;

int time_count(std::chrono::steady_clock::time_point tstart){
  return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - tstart).count();
}

void SpIOX::update_running_means(arma::mat& E_A, const arma::mat& A, bool pr){
  // vi_it is the internal iteration counter
  if(pr){
    // simple moving average
    if(vi_it > vi_min_iter){
      E_A = E_A + (A - E_A) / (vi_it - vi_min_iter + 1.0);  
    }
  } else {
    double gamma = 0.65;
    double alpha = vi_it < vi_min_iter ? 1 : 1.0/pow(vi_it - vi_min_iter + 1.0, gamma); // weight for present
    E_A = (1-alpha) * E_A + alpha * A;
  }
}

void SpIOX::init_theta_adapt(){
  // adaptive metropolis
  theta_mcmc_counter = 0;
  which_theta_elem = arma::zeros<arma::uvec>(0);
  arma::uvec oneuv = arma::ones<arma::uvec>(1);
  
  if(phi_sampling){
    which_theta_elem = arma::join_vert(which_theta_elem, 0*oneuv);
  }
  if(sigmasq_sampling){
    which_theta_elem = arma::join_vert(which_theta_elem, 1*oneuv);
  }
  if(nu_sampling){
    which_theta_elem = arma::join_vert(which_theta_elem, 2*oneuv);
  }
  if(alpha_sampling){
    which_theta_elem = arma::join_vert(which_theta_elem, 3*oneuv);
  }
  
  int n_theta_par = q * which_theta_elem.n_elem;
  
  arma::mat bounds_all = arma::zeros(4, 2); // make bounds for all, then subset
  bounds_all.row(0) = arma::rowvec({.3, 200}); // phi
  bounds_all.row(1) = arma::rowvec({1e-6, 100}); // sigma
  if(matern){
    bounds_all.row(2) = arma::rowvec({1e-5, 2.1}); // nu  
  } else {
    // power exponential
    bounds_all.row(2) = arma::rowvec({1, 2}); // nu
  }
  
  //bounds_all.row(3) = arma::rowvec({1e-16, 100}); // tausq
  // 1-alpha is the proportion of variance explained by spatial component
  // alpha is is the proportion explained by nugget effect
  bounds_all.row(3) = arma::rowvec({1e-10, 1-1e-10}); // alpha
  bounds_all = bounds_all.rows(which_theta_elem);
  theta_unif_bounds = arma::zeros(0, 2);
  
  for(int j=0; j<q; j++){
    theta_unif_bounds = arma::join_vert(theta_unif_bounds, bounds_all);
  }
  
  arma::mat theta_metrop_sd = 0.05 * arma::eye(n_theta_par, n_theta_par);
  theta_adapt = RAMAdapt(n_theta_par, theta_metrop_sd, 0.24);
  theta_adapt_active = true;
  
  if(q > 2){
    // conditional update for more than 2 outcomes
    c_theta_unif_bounds = bounds_all;
    int c_theta_par = which_theta_elem.n_elem;
    arma::mat c_theta_metrop_sd = 0.05 * arma::eye(c_theta_par, c_theta_par);
    c_theta_adapt = std::vector<RAMAdapt>(q);
    for(int j=0; j<q; j++){
      c_theta_adapt[j] = RAMAdapt(c_theta_par, c_theta_metrop_sd, 0.24);
    }
    // ---  
  }
  
}

void SpIOX::compute_V(){ 
  // V is made of
  // B (in the response model only)
  // theta, and cluster assignments
  
  // it is used in 
  // Sigma
  // gibbs for theta
  
  // whiten the residuals from spatial dependence
  if(latent_model>0){
    V = W;
  } else {
    V = YXB;
  }
  
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
  for(unsigned int j=0; j<q; j++){
    V.col(j) = daggps.at(j).H_times_A(V.col(j));
  }
  
  bool do_VTV = vi & (latent_model>0);
  if(do_VTV){
    VTV = V.t() * V; 
  }
}

void SpIOX::update_B(){
  if(latent_model==0){
    arma::vec daggp_logdets = arma::zeros(q);
    
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
    for (unsigned int j = 0; j < q; ++j) {
      daggp_logdets(j)  = daggps.at(j).precision_logdeterminant;
      Ytilde.col(j)     = daggps.at(j).H_times_A(Y.col(j));
      HX.slice(j)       = daggps.at(j).H_times_A(X);
    }
  
    arma::mat HX_mat(HX.memptr(), n, p * q, false, true);   // no-copy view of the cube
    arma::mat G = HX_mat.t() * HX_mat;                      // (qp x qp)
    
    arma::mat XtX(q * p, q * p);
    for (unsigned int a = 0; a < q; ++a) {
      for (unsigned int b = 0; b < q; ++b) {
        XtX.submat(a * p, b * p, (a + 1) * p - 1, (b + 1) * p - 1) =
          Q(a, b) *
          G.submat(a * p, b * p, (a + 1) * p - 1, (b + 1) * p - 1);
      }
    }
    
    arma::mat YS = Ytilde * Q;                          // n x q
    arma::vec XtY(q * p);
    for (unsigned int a = 0; a < q; ++a) {
      XtY.subvec(a * p, (a + 1) * p - 1) = HX.slice(a).t() * YS.col(a);
    }
    
    // Posterior precision and sample (if not vi)
    arma::mat post_precision = XtX;
    arma::vec vecB_Var = arma::vectorise(B_Var);
    post_precision.diag()   += 1.0 / vecB_Var;
    arma::mat L        = arma::chol(arma::symmatu(post_precision), "lower");
    arma::mat pp_ichol = arma::inv(arma::trimatl(L));
    Beta_UQ            = pp_ichol.t() * pp_ichol;
    
    arma::mat randnormat = (vi ? 0.0 : 1.0) * arma::randn(p * q);
    arma::vec beta       = Beta_UQ * XtY + pp_ichol.t() * randnormat;
    B = arma::mat(beta.memptr(), p, q);
    
  } else {
    // update B via gibbs for the latent model
    // btw we could make this into a conjugate MN update rather than conj N
    Ytilde = Y - W;
    arma::mat mvnorm = arma::randn(p, q);
    for(int j=0; j<q; j++){
      arma::mat X_available = X.rows(avail_by_outcome(j));
      arma::mat XtX = X_available.t() * X_available;
      arma::mat post_precision = arma::diagmat(1.0/B_Var.col(j)) + XtX/Ddiag(j);
      arma::mat pp_ichol = arma::inv(arma::trimatl(arma::chol(arma::symmatu(post_precision), "lower")));
      arma::vec yj = Ytilde.col(j);
      arma::vec y_available = yj.rows(avail_by_outcome(j));
      arma::vec XtYtildej = arma::trans(X_available) * y_available;
      
      arma::vec B_mean = pp_ichol.t() * pp_ichol * XtYtildej/Ddiag(j);
      B.col(j) = B_mean + pp_ichol.t() * mvnorm.col(j);
      
    }
  }
  
  YXB = Y - X*B;
  
}

void SpIOX::update_BW_asis(int& cg_iter, arma::mat& B, arma::mat& W, bool sampling){
  // Hold eta = XB + W constant; resample B then recover W = eta - XB.
  arma::mat eta = X * B + W;
  
  // whiten Yasis and X; HX[j] = H_j X 
  arma::mat Yasis(n, q);
  arma::vec daggp_logdets(q, arma::fill::zeros);
  
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
  for(unsigned int j = 0; j < q; ++j){
    Yasis.col(j)     = daggps.at(j).H_times_A(eta.col(j));
    HX.slice(j)      = daggps.at(j).H_times_A(X);
    daggp_logdets(j) = daggps.at(j).precision_logdeterminant;
  }
  
  //  Jacobi preconditioner 
  // P[j,j] diagonal at column c: Q(j,j) * ||HX[j].col(c)||^2 + 1/B_Var(c,j)
  arma::mat Mdiag_mat(p, q);
  for(unsigned int j = 0; j < q; ++j){
    arma::vec colsq(p);
    for(unsigned int c = 0; c < p; ++c){
      const arma::vec h = HX.slice(j).col(c);
      colsq(c) = arma::dot(h, h);
    }
    Mdiag_mat.col(j) = Q(j,j) * colsq + 1.0 / B_Var.col(j);
  }
  arma::vec Mdiag_vec = arma::vectorise(Mdiag_mat);
  
  const double diag_floor = 1e-12;
  for(arma::uword i = 0; i < Mdiag_vec.n_elem; ++i)
    if(!(Mdiag_vec(i) > diag_floor)) Mdiag_vec(i) = diag_floor;
  
  auto apply_Minv = [&](const arma::vec& r_in, arma::vec& z_out){
    z_out = r_in / Mdiag_vec;
  };
  
  
  // ----- Posterior precision multiply -----
  // (P b)[:,j] = HX[j]^T * (HX b * Q)[:,j] + (1/B_Var[:,j]) % b[:,j]
  arma::mat invBVar = 1.0 / B_Var;

  auto post_prec_mv = [&](const arma::vec& x_in, arma::vec& y_out){
    arma::mat Bin (const_cast<double*>(x_in.memptr()), p, q, false, true);
    arma::mat Bout(y_out.memptr(),                     p, q, false, true);
    
    arma::mat HXb(n, q);
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
    for(unsigned int j = 0; j < q; ++j) HXb.col(j) = HX.slice(j) * Bin.col(j);
    
    arma::mat HXbQ = HXb * Q;
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
    for(unsigned int j = 0; j < q; ++j)
      Bout.col(j) = HX.slice(j).t() * HXbQ.col(j) + invBVar.col(j) % Bin.col(j);
  };
  
  arma::mat YasisQ = Yasis * Q;
  
  arma::mat Wsamp, Z2;
  if(sampling){
    Wsamp = arma::randn(n, q) * Si.t();
    Z2 = arma::randn(p, q);
  } else {
    Wsamp = arma::zeros(n, q);
    Z2 = arma::zeros(p, q);
}

  arma::mat invSqrtBVar = 1.0 / arma::sqrt(B_Var);
  arma::mat RHS_mat(p, q);
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
  for(unsigned int j = 0; j < q; ++j){
    RHS_mat.col(j) = HX.slice(j).t() * (YasisQ.col(j) + Wsamp.col(j))
    + invSqrtBVar.col(j) % Z2.col(j);
  }
  arma::vec post_meansample = arma::vectorise(RHS_mat);

  // ----- PCG solve -----
  // Cold start (x0 = 0).  See gibbs_BW_block for why warm-starting from the
  // previous sample biases Bhattacharya draws when the preconditioner is
  // near-exact.
  arma::vec bbefore = arma::zeros<arma::vec>(B.n_elem);
  arma::vec b = pcg_mf(post_prec_mv, apply_Minv, cg_iter, post_meansample,
                       bbefore, 1e-6, pq, num_threads);

  B = arma::mat(b.memptr(), p, q);
  W = eta - X * B;
}

void SpIOX::gibbs_BW_block(int& cg_iter, PrecondChoice precond, bool sampling,
                           int cg_maxit_override){
  // per-entry inverse noise variance (zero at missing)
  arma::mat invD_mat(n, q, arma::fill::zeros);
  // arma::mat invSqrtD_mat(n, q, arma::fill::zeros);
  for(unsigned int j = 0; j < q; ++j){
    const double invDj  = 1.0 / Ddiag(j);
    // const double invSDj = 1.0 / std::sqrt(Ddiag(j));
    for(unsigned int i = 0; i < n; ++i){
      if(!missing_mat(i, j)){
        invD_mat(i, j)     = invDj;
        // invSqrtD_mat(i, j) = invSDj;
      }
    }
  }
  
  arma::mat Gscore = invD_mat % Y;
  arma::mat B_old = B;
  
  general_BW_block(invD_mat, Gscore, Q, Si,
                   cg_iter, precond, sampling,
                   cg_maxit_override, false);
  
  // Gaussian specific 
  YXB += X * (B_old - B);
}

void SpIOX::gibbs_w_block_precision(int& cg_iter, PrecondChoice precond,
                                    bool sampling, int cg_maxit_override){
  // W | B, Σ, θ, D  via precision-domain PCG with B held fixed at its current
  // value.  This is the W-only counterpart of gibbs_BW_block: identical
  // conditional W precision and identical W-half preconditioner, but B is NOT in
  // the system — it is sampled separately (conjugate B|W via update_B, then a
  // non-centred ASIS refresh).  Used by the blocked route (joint_BW = false).
  //
  //   conditional precision : P_WW = Λ_W + diag(invD)
  //     Λ_W w : Hw.col(j)=H_j w.col(j); HwQ=Hw·Q; out.col(i)=H_iᵀ HwQ.col(i)
  //   data term             : cW   = invD ⊙ (Y - XB)
  //   Bhattacharya noise    : prior η ~ N(0, Λ_W)  (blkdiag(Hᵀ)·vec(Z·Siᵀ)),
  //                           lik   ξ ~ N(0, D⁻¹)   (Z ⊙ invSqrtD).

  // per-entry inverse noise variance (zero at missing)
  arma::mat invD_mat(n, q, arma::fill::zeros);
  arma::mat invSqrtD_mat(n, q, arma::fill::zeros);
  for(unsigned int j = 0; j < q; ++j){
    const double invDj  = 1.0 / Ddiag(j);
    const double invSDj = 1.0 / std::sqrt(Ddiag(j));
    for(unsigned int i = 0; i < n; ++i){
      if(!missing_mat(i, j)){
        invD_mat(i, j)     = invDj;
        invSqrtD_mat(i, j) = invSDj;
      }
    }
  }

  const arma::uword Nw = n * q;

  // ----- conditional W precision multiply: P_WW w = Λ_W w + invD ⊙ w -----
  auto wprec_mv = [&](const arma::vec& x_in, arma::vec& y_out){
    arma::mat Win (const_cast<double*>(x_in.memptr()), n, q, false, true);
    arma::mat Wout(y_out.memptr(),                     n, q, false, true);
    arma::mat Hw(n, q);
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
    for(unsigned int j = 0; j < q; ++j) Hw.col(j) = daggps[j].H_times_A(Win.col(j));
    arma::mat HwQ = Hw * Q;
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
    for(unsigned int i = 0; i < q; ++i)
      Wout.col(i) = daggps[i].Ht_times_A(HwQ.col(i)) + invD_mat.col(i) % Win.col(i);
  };

  // ----- W-half preconditioner (mirrors gibbs_BW_block, B half dropped) -----
  arma::vec Mdiag_vec;   // JACOBI
  std::function<void(const arma::vec&, arma::vec&)> apply_Minv;

  if(precond == PRECOND_JACOBI){
    arma::mat H_col_sq(n, q);
    for(unsigned int i = 0; i < q; ++i) H_col_sq.col(i) = daggps[i].H_col_squared_norms();
    arma::mat Mdiag_W(n, q);
    for(unsigned int i = 0; i < q; ++i)
      Mdiag_W.col(i) = Q(i, i) * H_col_sq.col(i) + invD_mat.col(i);
    Mdiag_vec = arma::vectorise(Mdiag_W);
    const double diag_floor = 1e-12;
    for(arma::uword i = 0; i < Mdiag_vec.n_elem; ++i)
      if(!(Mdiag_vec(i) > diag_floor)) Mdiag_vec(i) = diag_floor;
    apply_Minv = [&](const arma::vec& r_in, arma::vec& z_out){ z_out = r_in / Mdiag_vec; };

  } else if(precond == PRECOND_VADU){  // cadence governed by cg_rebuild (default ALWAYS).
    if(pc_rebuild_now(PRECOND_VADU, vadu_pc_n_builds)){
      auto t_pc = std::chrono::steady_clock::now();
      build_vadu_Minv();   // per-location M_i^{-1} = (Q + diag_j R_i^{(j)}w_i^{(j)})^{-1}
      ++vadu_pc_n_builds;
      pc_build_seconds += time_count(t_pc) / 1e6;
    }
    apply_Minv = [&](const arma::vec& r_in, arma::vec& z_out){
      vadu_mv_apply(r_in.memptr(), z_out.memptr());   // H^{-1} M^{-1} H^{-ᵀ}
    };

  } else if(precond == PRECOND_POSTCOV){
    // Multivariate latent posterior-conditional: a SINGLE block-Vecchia factor of
    // the joint posterior covariance (q×q blocks couple all outcomes per location
    // via R_i = Λ_iΣΛ_i shrunk by the local datum).  No R_corr mix — the coupling
    // is inside the factor.  Decouples per outcome at diagonal Σ.  Default cadence ONCE.
    if(pc_rebuild_now(PRECOND_POSTCOV, postcov_n_builds)){
      auto t_pc = std::chrono::steady_clock::now();
      build_postcov_factors();
      pc_build_seconds += time_count(t_pc) / 1e6;
    }
    apply_Minv = [&](const arma::vec& r_in, arma::vec& z_out){
      postcov_apply(r_in.memptr(), z_out.memptr());
    };

  }

  // ----- RHS (Bhattacharya) -----
  arma::mat resid = Y - X * B;            // Y is 0 at missing; invD is 0 there
  arma::mat cW    = invD_mat % resid;
  arma::mat Unorm, Zlik_sc;
  if(sampling){
    Unorm = arma::randn(n, q) * Si.t();
    for(unsigned int j = 0; j < q; ++j) Unorm.col(j) = daggps[j].Ht_times_A(Unorm.col(j));
    Zlik_sc = arma::randn(n, q) % invSqrtD_mat;
  } else {
    Unorm   = arma::zeros(n, q);
    Zlik_sc = arma::zeros(n, q);
  }
  arma::vec rhs = arma::vectorise(cW + Unorm + Zlik_sc);

  arma::vec x0 = arma::zeros<arma::vec>(Nw);
  const int cg_maxit = (cg_maxit_override > 0) ? cg_maxit_override : static_cast<int>(nq);
  arma::vec sol = pcg_mf(wprec_mv, apply_Minv, cg_iter, rhs, x0,
                         5*1e-5, cg_maxit, num_threads);

  W = arma::mat(sol.memptr(), n, q);
}

void SpIOX::vi_Beta_UQ(){
  if(vi_it > vi_min_iter){
    arma::vec beta_v = arma::vectorise(B);
    
    delta_t = beta_v - beta_running_mean;
    beta_running_mean = beta_running_mean + 1.0/(1.0+vi_it-vi_min_iter) * delta_t;
    Beta_UQ = Beta_UQ + delta_t * arma::trans(beta_v - beta_running_mean);
  } else {
    beta_running_mean = arma::vectorise(B);
  }
}

bool SpIOX::upd_theta_metrop(){
  std::chrono::steady_clock::time_point tstart;
  std::chrono::steady_clock::time_point tend;
  int timed = 0;
  
  theta_adapt.count_proposal();
  
  arma::vec phisig_cur = arma::vectorise( theta.rows(which_theta_elem) );
  
  Rcpp::RNGScope scope;
  arma::vec U_update = arma::randn(phisig_cur.n_elem);
  
  arma::vec phisig_alt = par_huvtransf_back(par_huvtransf_fwd(
    phisig_cur, theta_unif_bounds) + 
      theta_adapt.paramsd * U_update, theta_unif_bounds);
  
  arma::mat theta_alt = theta;
  arma::mat phisig_alt_mat = arma::mat(phisig_alt.memptr(), which_theta_elem.n_elem, q);
  
  theta_alt.rows(which_theta_elem) = phisig_alt_mat; 
  
  if(!theta_alt.is_finite()){
    Rcpp::stop("Some value of theta outside of MCMC search limits.\n");
  }
  
  // ---------------------
  // by default this runs when q=1 or q=2, no need for omp
  // create proposal daggp
  // this can run in parallel but update_theta already uses omp
  // do not run this in parallel, will be faster this way
  tstart = std::chrono::steady_clock::now();
  for(unsigned int i=0; i<q; i++){
    daggps_alt[i].update_theta(theta_alt.col(i));
  }
  tend = std::chrono::steady_clock::now();
  timed = std::chrono::duration_cast<std::chrono::microseconds>(tend - tstart).count();
  //Rcpp::Rcout << "update dag proposal: " << timed << endl;
  // ----------------------
  // current density and proposal density
  tstart = std::chrono::steady_clock::now();
  
  arma::vec daggp_logdets = arma::zeros(q);
  arma::mat V_alt = V;
  arma::vec daggp_alt_logdets = arma::zeros(q);
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
  for(unsigned int j=0; j<q; j++){
    arma::mat Target;
    if(latent_model>0){
      Target = W.col(j);
    } else {
      Target = YXB.col(j);
    }
    V_alt.col(j) = daggps_alt.at(j).H_times_A(Target);// * (Y.col(j) - X * B.col(j));
    daggp_logdets(j) = daggps.at(j).precision_logdeterminant;
    daggp_alt_logdets(j) = daggps_alt.at(j).precision_logdeterminant;
  }
  tend = std::chrono::steady_clock::now();
  timed = std::chrono::duration_cast<std::chrono::microseconds>(tend - tstart).count();
  //Rcpp::Rcout << "computing V: " << timed << endl;
  
  tstart = std::chrono::steady_clock::now();
  // current
  //arma::mat Ytildemat = arma::mat(vecYtilde.memptr(), n, q, false, true);
  //arma::vec ytilde = arma::vectorise(V * Si);
  double curr_ldet = +0.5*arma::accu(daggp_logdets);
  double curr_logdens = curr_ldet - 0.5*arma::accu(pow(V * Si, 2.0));
  
  // proposal
  //arma::mat Ytildemat_alt = arma::mat(vecYtilde_alt.memptr(), n, q, false, true);
  //arma::vec ytilde_alt = arma::vectorise(V_alt * Si);
  double prop_ldet = +0.5*arma::accu(daggp_alt_logdets);
  double prop_logdens = prop_ldet - 0.5*arma::accu(pow(V_alt * Si, 2.0));
  
  tend = std::chrono::steady_clock::now();
  timed = std::chrono::duration_cast<std::chrono::microseconds>(tend - tstart).count();
  //Rcpp::Rcout << "computing VSi: " << timed << endl;
  
  // priors
  double logpriors = 0;
  for(unsigned int j=0; j<q; j++){
    if(sigmasq_sampling){
      logpriors += invgamma_logdens(theta_alt(1,j), 2, 1) - invgamma_logdens(theta(1,j), 2, 1);
    }
    if(alpha_sampling){
      // nned to change this prior if alpha is a proportion which we expect ~0
      //logpriors += expon_logdens(theta_alt(3,j), 25) - expon_logdens(theta(3,j), 25);
    }
  }
  
  // ------------------
  // make move
  double jacobian  = calc_jacobian(phisig_alt, phisig_cur, theta_unif_bounds);
  double logaccept = prop_logdens - curr_logdens + jacobian + logpriors;
  
  bool accepted = do_I_accept(logaccept);
  
  if(accepted){
    theta = theta_alt;
    std::swap(daggps, daggps_alt);
    std::swap(V, V_alt);
  } 
  
  theta_adapt.update_ratios();
  
  if(theta_adapt_active){
    theta_adapt.adapt(U_update, exp(logaccept), theta_mcmc_counter); 
  }
  
  theta_mcmc_counter++;
  
  return accepted;
}

arma::uvec SpIOX::upd_theta_metrop_conditional(){
  
  using namespace std::chrono;
  auto t_start = steady_clock::now();
  long t_proposal = 0; // update dags with new theta proposals beforehand
  long t_mcmc  = 0; // sequential portion
  
  arma::uvec oneuv = arma::ones<arma::uvec>(1);
  arma::uvec accepteds = arma::zeros<arma::uvec>(q);
  
  // we can prepare the proposals beforehand, then only do the strictly sequential part
  
  Rcpp::RNGScope scope;
  arma::mat U_update = arma::randn(which_theta_elem.n_elem, q);
  arma::mat phisig_alt = arma::zeros(which_theta_elem.n_elem, q);
  arma::mat phisig_cur = arma::zeros(which_theta_elem.n_elem, q);
  arma::mat theta_alt = theta; // proposals
  
  for(int j=0; j<q; j++){
    c_theta_adapt[j].count_proposal();
    phisig_cur.col(j) = theta(which_theta_elem, oneuv*j);
    
    phisig_alt.col(j) = par_huvtransf_back(par_huvtransf_fwd(
      phisig_cur.col(j), c_theta_unif_bounds) + 
        c_theta_adapt[j].paramsd * U_update.col(j), c_theta_unif_bounds);
  
    // proposal for theta matrix
    //arma::mat theta_alt = theta;
    theta_alt(which_theta_elem, oneuv*j) = phisig_alt.col(j); 
    
    if(!theta_alt.is_finite()){
      Rcpp::stop("Some value of theta outside of MCMC search limits.\n");
    }
  }
  auto t0 = steady_clock::now();
  
  if(gridded){
    // gridded data -- 
#ifdef _OPENMP
#pragma omp parallel for num_threads(q) //***
#endif
    for(int j=0; j<q; j++){
      // create proposal daggp // 1 thread here
      daggps_alt[j].update_theta(theta_alt.col(j));
    }
  } else {
    for(int j=0; j<q; j++){
      // create proposal daggp // omp inside here -- its slightly faster
      daggps_alt[j].update_theta(theta_alt.col(j));
    }
  }
  

  
  auto t1 = steady_clock::now();
  t_proposal += duration_cast<microseconds>(t1 - t0).count();
  
  // sequential
  for(int j=0; j<q; j++){
    // conditional density of Y_j | Y_-j (or W depending on target)
    arma::mat V_alt = V;
    
    if(latent_model>0){
      V_alt.col(j) = daggps_alt.at(j).H_times_A(W.col(j));// * (Y.col(j) - X * B.col(j));
    } else {
      V_alt.col(j) = daggps_alt.at(j).H_times_A(YXB.col(j));// * (Y.col(j) - X * B.col(j));
    }
    
    double c_daggp_logdet = daggps.at(j).precision_logdeterminant;
    double c_daggp_alt_logdet = daggps_alt.at(j).precision_logdeterminant;
    
    arma::vec Vjc = arma::zeros(n);
    arma::vec Vjc_alt = arma::zeros(n);
    for(int jc=0; jc<q; jc++){
      Vjc += Q(j, jc)/Q(j,j) * V.col(jc);
      Vjc_alt += Q(j, jc)/Q(j,j) * V_alt.col(jc);
    }
    
    double core_alt = arma::accu(pow(Vjc_alt, 2.0)); 
    double core = arma::accu(pow(Vjc, 2.0)); 
    double prop_logdens = 0.5 * c_daggp_alt_logdet - Q(j,j)/2.0 * core_alt;
    double curr_logdens = 0.5 * c_daggp_logdet - Q(j,j)/2.0 * core;
    
    // priors
    double logpriors = 0;
    if(sigmasq_sampling){
      logpriors += invgamma_logdens(theta_alt(1,j), 2, 1) - invgamma_logdens(theta(1,j), 2, 1);
    }
    if(alpha_sampling){
      //logpriors += expon_logdens(theta_alt(3,j), 25) - expon_logdens(theta(3,j), 25);
    }
    
    // ------------------
    // make move
    double jacobian  = calc_jacobian(phisig_alt.col(j), phisig_cur.col(j), c_theta_unif_bounds);
    double logaccept = prop_logdens - curr_logdens + jacobian + logpriors;
    
    accepteds(j) = do_I_accept(logaccept);
    
    if(accepteds(j)){
      theta.col(j) = theta_alt.col(j);
      std::swap(daggps.at(j), daggps_alt.at(j));
      //std::swap(V, V_alt);
      V.col(j) = V_alt.col(j);
    } 
    
    c_theta_adapt[j].update_ratios();
    
    if(theta_adapt_active){
      c_theta_adapt[j].adapt(U_update.col(j), exp(logaccept), theta_mcmc_counter); 
    }
    
    theta_mcmc_counter++;
  }
  
  auto t2 = steady_clock::now();
  t_mcmc += duration_cast<microseconds>(t2 - t1).count();
  
  // --- PRINT RESULTS ---
  //Rcpp::Rcout << "--- Conditional Update Timing (microseconds) ---" << std::endl;
  //Rcpp::Rcout << "Proposal build:   " << t_proposal << std::endl;
  //Rcpp::Rcout << "MCMC seq:      " << t_mcmc << std::endl;
  
  return accepteds;
}

void SpIOX::cache_blanket_comps(const arma::uvec& theta_changed){
  // Inner-product structure of H_r and H_s over markov blankets, computed
  // once per theta change.  Used by w_sequential_singlesite (latent_model=2)
  // and missing-data imputation in the response model.
  //
  // For each location i:
  //   Rw_no_Q(ix)(r, s)                       = <col_i(H_r), col_i(H_s)>
  //   Pblanket_no_Q(ix)(r, s*mbsize + k)      = <col_i(H_r), col_{blanket(k)}(H_s)>
  //
  // The original implementation materialised q arma::sp_mat copies via
  // make_H() (one per outcome) and then walked sparse columns through arma's
  // CSC machinery.  Here we go directly to the col-major Eigen mirror that
  // DagGP already keeps (H_eigen), and compute every required sparse-column
  // inner product via paired InnerIterators — no arma::sp_mat copy, no
  // intermediate Hitt / Hblanket allocation per location.
  int nfill = latent_model == 2 ? n : rows_with_missing.n_elem;

  // Sparse-sparse column dot product on col-major Eigen sparse matrices.
  // O(min(nnz(colA), nnz(colB))) — for Vecchia, both columns have ~m+1 entries
  // (the diagonal at i + the rows where i appears as a parent), so this is
  // cheap.  daggps[r] and daggps[s] share the same DAG, so col-i sparsity
  // patterns match exactly; the merge logic below is kept for safety.
  auto dot_cols = [](const Eigen::SparseMatrix<double>& A, int colA,
                     const Eigen::SparseMatrix<double>& B, int colB) -> double {
    double acc = 0.0;
    Eigen::SparseMatrix<double>::InnerIterator ia(A, colA), ib(B, colB);
    while (ia && ib) {
      if (ia.row() == ib.row()) { acc += ia.value() * ib.value(); ++ia; ++ib; }
      else if (ia.row() < ib.row())                                ++ia;
      else                                                         ++ib;
    }
    return acc;
  };

#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
  for(int ix=0; ix<nfill; ix++){
    int i = latent_model == 2 ? ix : rows_with_missing(ix);

    // Assume all daggps share the same DAG (true by construction; mblanket
    // depends only on DAG topology, not on theta).
    const arma::uvec& mblanket = daggps[0].mblanket(i);
    const int mbsize = mblanket.n_elem;

    Rw_no_Q(ix)        = arma::zeros(q, q);
    Pblanket_no_Q(ix)  = arma::zeros(q, q*mbsize);

    // Rw_no_Q : symmetric q×q of inner products of col i across outcomes.
    for(int r = 0; r < (int)q; ++r){
      for(int s = 0; s <= r; ++s){
        const double v = dot_cols(daggps[r].H_eigen, i,
                                  daggps[s].H_eigen, i);
        Rw_no_Q(ix)(r, s) = v;
        if (s != r) Rw_no_Q(ix)(s, r) = v;
      }
    }

    // Pblanket_no_Q : block-q of q×mbsize inner-product matrices, one per s.
    //   Pblanket_no_Q(ix)(r, s*mbsize + k) = <col_i(H_r), col_{mblanket(k)}(H_s)>
    for(int s = 0; s < (int)q; ++s){
      const int col_base = s * mbsize;
      for(int k = 0; k < mbsize; ++k){
        const int col_idx = (int)mblanket(k);
        for(int r = 0; r < (int)q; ++r){
          Pblanket_no_Q(ix)(r, col_base + k) =
            dot_cols(daggps[r].H_eigen, i,
                     daggps[s].H_eigen, col_idx);
        }
      }
    }
  }
}

void SpIOX::w_sequential_singlesite(const arma::uvec& theta_changed){
  double ms_if_cache = 0;
  double ms_omp_for = 0;
  double ms_sample = 0;
  
  // precompute stuff in parallel so we can do fast sequential sampling after
  
  arma::field<arma::mat> Hw(n);
  arma::field<arma::mat> Rw(n);
  
  arma::cube Postcov = arma::zeros(q, q, n);
  arma::mat randcomp = arma::randn(q, n);
  //arma::mat mvnorm = arma::randn(q, n);
  
  // V = whitened Y-XB or W
  if(arma::any(theta_changed != 0)){
    // perform this update if theta has changed and we need to recompute the
    // GP-related matrices that depend on it
    cache_blanket_comps(theta_changed);
  }
  
  
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
  for(int i=0; i<n; i++){
    Rw(i) = Q % Rw_no_Q(i);
    
    // assume all the same dag otherwise we go cray
    arma::uvec mblanket = daggps[0].mblanket(i);
    int mbsize = mblanket.n_elem;
    arma::mat Pblanket = arma::zeros(q, q*mbsize);
    arma::mat Di_obs = arma::zeros(q,q);
    
    for(int j = 0; j < q; j++) {
      int startcol = j * mbsize;
      int endcol = (j + 1) * mbsize - 1;
      Pblanket.cols(startcol, endcol) = arma::diagmat(Q.col(j)) * Pblanket_no_Q(i).cols(startcol, endcol);
      if(!missing_mat(i,j)){
        Di_obs(j,j) = 1.0/Ddiag(j);
      }
    }
    
    Hw(i) = - Pblanket;
    arma::mat invcholP = arma::inv(arma::trimatl(arma::chol(Rw(i) + Di_obs, "lower")));
    Postcov.slice(i) = invcholP.t() * invcholP;
    randcomp.col(i) = invcholP.t() * randcomp.col(i);
  }

  // visit every location and sample from latent effects 
  // conditional on data and markov blanket
  for(int c=0; c < daggps[0].colors.n_elem; c++){
    arma::uvec nodes_in_color = daggps[0].colors(c);
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
    for(int ix=0; ix < nodes_in_color.n_elem; ix++){
      int i = nodes_in_color(ix);
      //for(int i=0; i<n; i++){
      // data contributions may be null if data missing
      arma::vec Di_YXB = arma::zeros(q);
      for(int j=0; j<q; j++){
        if(!missing_mat(i,j)){
          Di_YXB(j) = YXB(i,j)/Ddiag(j);
        }
      }
      
      arma::uvec mblanket = daggps[0].mblanket(i);
    
      // sample
      arma::vec W_mean = Postcov.slice(i) * ( Hw(i) * arma::vectorise( W.rows(mblanket) ) + Di_YXB );
      W.row(i) = arma::trans(  W_mean + randcomp.col(i) );
      W_RB.row(i) = W_mean.t();
      
      if(W.has_nan()){
        Rcpp::stop("Found nan in W.\n");
      }
    }
  }


}

void SpIOX::gibbs_w_sequential_byoutcome(int& cg_iter, PrecondChoice precond,
                                         int cg_maxit_override){
  // Per-outcome Gibbs update of W.col(j) | W_{-j}, B, Σ, Ddiag, Y.
  //
  // Conditional model (Gaussian conditional from the IOX latent prior):
  //   prior:   W_j | W_{-j} ~ N( μ_j|-j,  K_j|-j )
  //            K_j|-j = (1 / Q_jj) · C_j   (C_j = Vecchia cov for outcome j)
  //            μ_j|-j = -(1/Q_jj) · H_j^{-1} · (V_{-j} · Q_{-j,j})
  //                     where V_k = H_k W_k (whitened outcomes).
  //   obs:     YXB_j = W_j + ε,  ε ~ N(0, D_j I)   (D_j = Ddiag(j))
  //
  // Operator (precision-form, per outcome):
  //   A_j = Q_jj · H_jᵀ H_j + diag(invD_j)            (n×n SPD)
  //
  // Bhattacharya precision-form RHS:
  //   b = invD ⊙ YXB − H_jᵀ (V_{-j} · Q_{-j,j})
  //     + sqrt(Q_jj) · H_jᵀ · z_prior   (z_prior ~ N(0, I_n))
  //     + invSqrtD ⊙ z_lik              (z_lik   ~ N(0, I_n) zeroed at missing)
  //
  // Solved by matrix-free PCG.  The supported preconditioners use Eigen ops via
  // daggps[j] — no arma::sp_mat is materialised here.
  //
  // PRECOND_JACOBI : M^{-1} = diag(A_j)^{-1}
  //                  diag(A_j)(k) = Q_jj · ||H_j.col(k)||² + invD(k)
  // PRECOND_VADU   : per-outcome Vecchia-diagonal-update factor (see below).

  // POSTCOV (the joint multivariate factor) has no cross-outcome term to exploit
  // here — the sampling=3 operator A_j is already per-outcome — so it falls back to
  // VADU.  Build (or rebuild) the per-outcome VADU factor honouring cg_rebuild.
  if(precond == PRECOND_POSTCOV) precond = PRECOND_VADU;
  if(precond == PRECOND_VADU){
    // Per-outcome VADU: P_VADU,j^{-1} r = H_j^{-1}[ (H_j^{-T} r) / dscale_j^2 ],
    // dscale_j = sqrt(R_j ⊙ invD_j + Q_jj), R_j = sqrtR_j^2 (the per-location
    // Vecchia conditional variance of C_j).  No cross-outcome Σ-mix (R_corr = I)
    // since the sampling=3 operator A_j is already per-outcome.  Cadence honours
    // cg_rebuild (default ONCE under the simplified UI; "always" tracks Σ/Ddiag).
    if(pc_rebuild_now(PRECOND_VADU, vadu_pc_n_builds)){
      auto t_pc = std::chrono::steady_clock::now();
      bw_vadu_dscale.assign(q, arma::vec());
      for(int j = 0; j < (int)q; ++j){
        arma::vec invDj_vec(n, arma::fill::zeros);
        for(int i = 0; i < (int)n; ++i)
          if(!missing_mat(i, j)) invDj_vec(i) = 1.0 / Ddiag(j);
        arma::vec Rj = arma::square(daggps[j].sqrtR);
        bw_vadu_dscale[j] = arma::sqrt(Rj % invDj_vec + Q(j, j));
      }
      ++vadu_pc_n_builds;
      pc_build_seconds += time_count(t_pc) / 1e6;
    }
  }

  const bool has_missing = Y_needs_filling;

  arma::mat urands = arma::randn(n, q);
  arma::mat vrands = arma::randn(n, q);
  arma::uvec r1q = arma::regspace<arma::uvec>(0, q-1);

  YXB.elem(find(missing_mat)).zeros();

  cg_iter = 0;
  int cg_iter_sum = 0;

  for(int j = 0; j < (int)q; ++j){
    // per-location diagonal noise contribution for this column j
    arma::vec invD(n, arma::fill::zeros);
    arma::vec invSqrtD(n, arma::fill::zeros);
    for(int i = 0; i < (int)n; ++i){
      if(!missing_mat(i, j)){
        invD(i)      = 1.0 / Ddiag(j);
        invSqrtD(i)  = 1.0 / std::sqrt(Ddiag(j));
      }
    }

    arma::uvec notj = arma::find(r1q != j);
    arma::uvec jx   = arma::zeros<arma::uvec>(1) + j;

    const double Qjj     = Q(j, j);
    const double sqrtQjj = std::sqrt(Qjj);

    // Matrix-free operator A_j v = Q_jj · H_jᵀ (H_j v) + invD ⊙ v
    auto A_matvec = [&](const arma::vec& v_in, arma::vec& y_out){
      arma::vec Hv = daggps[j].H_times_A(v_in);
      y_out = Qjj * daggps[j].Ht_times_A(Hv);
      for(int i = 0; i < (int)n; ++i){
        if(!missing_mat(i, j)) y_out(i) += invD(i) * v_in(i);
      }
    };

    // PC apply
    arma::vec Mdiag;                                       // JACOBI: lives at outer scope so the lambda's reference stays valid
    std::function<void(const arma::vec&, arma::vec&)> apply_Minv;
    if(precond == PRECOND_VADU){
      // Per-outcome VADU: z = H_j^{-1}[ (H_j^{-T} r) / dscale_j^2 ].
      apply_Minv = [&, j](const arma::vec& r_in, arma::vec& z_out){
        arma::vec t = daggps[j].Ht_solve_A(r_in) / bw_vadu_dscale[j];
        z_out = daggps[j].H_solve_A(t / bw_vadu_dscale[j]);
      };
    } else {  // PRECOND_JACOBI
      Mdiag = Qjj * daggps[j].H_col_squared_norms() + invD;
      const double diag_floor = 1e-12;
      for(arma::uword k = 0; k < Mdiag.n_elem; ++k){
        if(!(Mdiag(k) > diag_floor)) Mdiag(k) = diag_floor;
      }
      apply_Minv = [&Mdiag](const arma::vec& r_in, arma::vec& z_out){
        z_out = r_in / Mdiag;
      };
    }

    // RHS pieces (all via Eigen Ht_times_A — no arma::sp_mat needed)
    arma::vec data_term(n);
    for(int i = 0; i < (int)n; ++i){
      data_term(i) = invD(i) * YXB(i, j);                  // 0 at missing
    }
    arma::vec prior_mean_term = - daggps[j].Ht_times_A(V.cols(notj) * Q.submat(notj, jx));
    arma::vec noise_prior     = sqrtQjj * daggps[j].Ht_times_A(vrands.col(j));
    arma::vec noise_lik(n, arma::fill::zeros);
    for(int i = 0; i < (int)n; ++i){
      if(!missing_mat(i, j)) noise_lik(i) = invSqrtD(i) * urands(i, j);
    }
    arma::vec rhs = data_term + prior_mean_term + noise_prior + noise_lik;

    // PCG solve (warm-started from current W.col(j))
    int it_j = 0;
    arma::vec x0 = W.col(j);
    const int maxit_j = (cg_maxit_override > 0) ? std::min((int)n, cg_maxit_override) : (int)n;
    W.col(j) = pcg_mf(A_matvec, apply_Minv, it_j, rhs, x0, 1e-5, maxit_j, num_threads);
    cg_iter_sum += it_j;

    V.col(j) = daggps[j].H_times_A(W.col(j));              // keep V in sync for next j
  }

  cg_iter = cg_iter_sum;
}


// One-time DAG-only precompute for postcov (see spiox.h postcov_*): the level
// schedule (for the parallel block solve), the children adjacency (for the back-solve
// gather), and the preallocated block storage + apply buffer.  Independent of θ/Σ/Ddiag,
// so it runs once and is reused across every rebuild and apply.
void SpIOX::postcov_setup(){
  // levels: level[i] = 1 + max level over parents (0 for roots).  daggps[0].dag(i)
  // holds indices < i (DAG order), so a single forward pass suffices.
  arma::uvec level(n, arma::fill::zeros);
  arma::uword maxlev = 0;
  for(int i = 0; i < (int)n; ++i){
    const arma::uvec& par = daggps[0].dag(i);
    arma::uword lev = 0;
    for(arma::uword t = 0; t < par.n_elem; ++t) lev = std::max(lev, level(par(t)) + 1);
    level(i) = lev;
    maxlev = std::max(maxlev, lev);
  }
  const arma::uword nlev = maxlev + 1;
  // counting sort of locations by level -> order, level_ptr
  arma::uvec cnt(nlev, arma::fill::zeros);
  for(int i = 0; i < (int)n; ++i) cnt(level(i))++;
  postcov_level_ptr.set_size(nlev + 1);
  postcov_level_ptr(0) = 0;
  for(arma::uword L = 0; L < nlev; ++L)
    postcov_level_ptr(L + 1) = postcov_level_ptr(L) + cnt(L);
  postcov_order.set_size(n);
  arma::uvec fill = postcov_level_ptr.head(nlev);   // running write cursor per level
  for(int i = 0; i < (int)n; ++i) postcov_order(fill(level(i))++) = (arma::uword)i;

  // children adjacency: invert the DAG (for each k and parent-position t, register
  // (k, t) under that parent) — needed for the race-free gather-form back-solve.
  std::vector<std::vector<arma::uword>> ck(n), ct(n);
  for(int k = 0; k < (int)n; ++k){
    const arma::uvec& par = daggps[0].dag(k);
    for(arma::uword t = 0; t < par.n_elem; ++t){
      ck[par(t)].push_back((arma::uword)k);
      ct[par(t)].push_back(t);
    }
  }
  postcov_child_k.set_size(n);
  postcov_child_t.set_size(n);
  for(int i = 0; i < (int)n; ++i){
    postcov_child_k(i) = arma::uvec(ck[i]);
    postcov_child_t(i) = arma::uvec(ct[i]);
  }

  // preallocate dense block storage (overwritten in place by every rebuild) + buffers
  postcov_L.assign(n, arma::mat(q, q, arma::fill::zeros));
  postcov_E.assign(n, arma::mat(q, q, arma::fill::zeros));
  postcov_D.assign(n, arma::mat());
  for(int i = 0; i < (int)n; ++i)
    postcov_D[i].set_size(q, daggps[0].dag(i).n_elem);   // col t = d_i^{(t)} (q-vector)
  postcov_buf.set_size(q, n);
  postcov_g.set_size(q, n);
  // per-thread apply scratch (avoids malloc churn / contention in the parallel solve)
  const int nt = std::max(1, num_threads);
  postcov_acc.assign(nt, arma::vec(q));
  postcov_u.assign(nt, arma::vec(q));

  postcov_setup_done = true;
}

// Multivariate POSTCOV: (re)compute the dense q×q blocks of the factor U.  Per
// location i, reusing b_i^{(j)} = daggps[j].h(i), sqrtR_i^{(j)} = daggps[j].sqrtR(i)
// (shared parents daggps[0].dag(i)), coupled across outcomes by Σ via Q = Σ^{-1}:
//     R_i^{-1} = Λ_i^{-1} Q Λ_i^{-1}            (Λ_i = diag_j sqrtR_i^{(j)}; no inverse)
//     K_i = R_i^{-1} + Ω_i,  F_i = K_i^{-1} = L_i L_iᵀ
//     postcov_L[i] = L_i ;  parent block -E_i D_i^{(t)} stored as
//     postcov_E[i] = L_iᵀΛ_i^{-1}Q (= L_iᵀR_i^{-1} with cols ·sqrtR) and
//     postcov_D[i] col t = b_i^{(t)}/sqrtR_i  (the diagonal of D_i^{(t)}).
// Storage is preallocated by postcov_setup() and overwritten in place (cheap
// rebuilds); rows are independent so the loop is OpenMP-parallel.
void SpIOX::build_postcov_factors(){
  if(!postcov_setup_done) postcov_setup();

#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
  for(int i = 0; i < (int)n; ++i){
    arma::vec sd(q), inv_sd(q);
    for(int j = 0; j < (int)q; ++j){ sd(j) = daggps[j].sqrtR(i); inv_sd(j) = 1.0 / sd(j); }
    arma::mat Rinv = Q;                                  // R_i^{-1} = Λ^{-1} Q Λ^{-1}
    Rinv.each_col() %= inv_sd;
    Rinv.each_row() %= inv_sd.t();
    arma::mat K = Rinv;                                  // K_i = R_i^{-1} + Ω_i
    for(int j = 0; j < (int)q; ++j)
      if(!missing_mat(i, j)) K(j, j) += 1.0 / Ddiag(j);
    arma::mat F = arma::inv_sympd(arma::symmatu(K));     // F_i = K_i^{-1} = L_i L_iᵀ
    postcov_L[i] = arma::chol(arma::symmatu(F), "lower");   // in place (q×q preallocated)

    const arma::uvec& par = daggps[0].dag(i);
    const arma::uword mi  = par.n_elem;
    if(mi == 0) continue;
    // Parent-t block -L_i^{-1}G_i^{(t)} = -E_i D_i^{(t)} with E_i location-only, D_i^{(t)} diagonal.
    // E_i = (L_iᵀ R_i^{-1}) Λ_i = L_iᵀ Λ_i^{-1} Q  (scale columns of L_iᵀR_i^{-1} by sqrtR).
    postcov_E[i] = postcov_L[i].t() * Rinv;        // L_iᵀ R_i^{-1}        (in place, q×q)
    postcov_E[i].each_row() %= sd.t();                // · Λ_i  ⇒  L_iᵀ Λ_i^{-1} Q
    for(arma::uword t = 0; t < mi; ++t)                  // d_i^{(t)} = b_i^{(t)} / sqrtR_i
      for(int j = 0; j < (int)q; ++j)
        postcov_D[i](j, t) = daggps[j].h(i)(t) * inv_sd(j);
  }
  ++postcov_n_builds;
}

// Apply M^{-1} = U^{-1} U^{-ᵀ} to one W-block (r_w / z_w length nq, outcome-major),
// LEVEL-SCHEDULED so the block substitution is OpenMP-parallel over locations rather
// than serial along the DAG.  Uses the factored parent block -E_i D_i^{(t)} (E_i q×q,
// D_i^{(t)} diagonal): the parent/child gather is a cheap diagonal weighting (O(mq)) and
// each location does ONE q×q gemv (E_i in fwd, E_iᵀ in back) instead of m.  Two q×n buffers:
//   back (Uᵀ s = r): levels DOWN; acc = r_i + Σ_child d_k^{(t)}⊙g_k; s_i = L_iᵀacc; g_i = E_iᵀs_i
//   fwd  (U  z = s): levels UP;   u = Σ_t d_i^{(t)}⊙z_par(t); s_i = L_i(s_i + E_i u)
// g_i is precomputed when s_i is finalised — children sit at higher levels, so g_k is ready
// before any parent reads it.  Each level is its own `omp parallel for`; a single region
// spanning both passes was measured *slower* — libgomp reuses the team across the repeated
// dispatches, and the residual parallel ceiling is the DAG's ~120-level critical path.
void SpIOX::postcov_apply(const double* r_w, double* z_w){
  arma::mat Rmat(const_cast<double*>(r_w), n, q, false, true);
  postcov_buf = Rmat.t();                    // q×n location-major (col i = r_i)
  arma::mat& s = postcov_buf;
  arma::mat& g = postcov_g;                  // back-pass: g_i = E_iᵀ s_i (i's parents read it)
  const int nlev = (int)postcov_level_ptr.n_elem - 1;

  for(int L = nlev - 1; L >= 0; --L){          // back-solve Uᵀ s = r
    const arma::uword lo = postcov_level_ptr(L), hi = postcov_level_ptr(L + 1);
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
    for(arma::uword idx = lo; idx < hi; ++idx){
#ifdef _OPENMP
      const int tid = omp_get_thread_num();
#else
      const int tid = 0;
#endif
      const arma::uword i = postcov_order(idx);
      arma::vec& acc = postcov_acc[tid];
      acc = s.col(i);                                              // r_i (own slot)
      const arma::uvec& ck = postcov_child_k(i);
      const arma::uvec& ct = postcov_child_t(i);
      for(arma::uword c = 0; c < ck.n_elem; ++c){                  // + Σ d_k^{(t)} ⊙ g_k
        const double* dcol = postcov_D[ck(c)].colptr(ct(c));    //   (= -B_{k,t}ᵀ s_k)
        const double* gcol = g.colptr(ck(c));
        for(int r = 0; r < (int)q; ++r) acc(r) += dcol[r] * gcol[r];
      }
      s.col(i) = postcov_L[i].t() * acc;                        // L_iᵀ
      if(daggps[0].dag(i).n_elem)                                  // parents will read g_i
        g.col(i) = postcov_E[i].t() * s.col(i);                 // g_i = E_iᵀ s_i
    }
  }

  for(int L = 0; L < nlev; ++L){               // forward-solve U z = s
    const arma::uword lo = postcov_level_ptr(L), hi = postcov_level_ptr(L + 1);
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
    for(arma::uword idx = lo; idx < hi; ++idx){
#ifdef _OPENMP
      const int tid = omp_get_thread_num();
#else
      const int tid = 0;
#endif
      const arma::uword i = postcov_order(idx);
      arma::vec& acc = postcov_acc[tid];
      acc = s.col(i);                                              // s_i (own slot)
      const arma::uvec& par = daggps[0].dag(i);
      if(par.n_elem){
        arma::vec& u = postcov_u[tid];                          // u = Σ_t d_i^{(t)} ⊙ z_par(t)
        u.zeros();
        for(arma::uword t = 0; t < par.n_elem; ++t){
          const double* dcol = postcov_D[i].colptr(t);
          const double* zcol = s.colptr(par(t));
          for(int r = 0; r < (int)q; ++r) u(r) += dcol[r] * zcol[r];
        }
        acc += postcov_E[i] * u;                                // one gemv (+E_i u)
      }
      s.col(i) = postcov_L[i] * acc;                            // L_i
    }
  }

  // transpose straight into the caller's z_w view (no Zout temp / std::copy)
  arma::mat Zout(z_w, n, q, false, true);
  Zout = s.t();                                 // n×q (outcome-major)
}

// Multivariate VADU: per-location q×q inverse of M_i = Q + diag_j(R_i^{(j)} w_i^{(j)})
// (R_i^{(j)} = sqrtR_i^{(j)2}, w_i^{(j)} = invD_ij, 0 at missing).  M_i is SPD (Q SPD +
// nonneg diagonal); rows independent → OpenMP-parallel.  See spiox.h bw_vadu_Minv.
void SpIOX::build_vadu_Minv(){
  bw_vadu_Minv.assign(n, arma::mat());
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
  for(int i = 0; i < (int)n; ++i){
    arma::mat Mi = Q;
    for(int j = 0; j < (int)q; ++j){
      const double Rij = daggps[j].sqrtR(i) * daggps[j].sqrtR(i);     // R_i^{(j)}
      const double wij = missing_mat(i, j) ? 0.0 : 1.0 / Ddiag(j);    // w_i^{(j)}
      Mi(j, j) += Rij * wij;
    }
    bw_vadu_Minv[i] = arma::inv_sympd(arma::symmatu(Mi));
  }
}

// Apply P_VADU^{-1} = H^{-1} M^{-1} H^{-ᵀ} to one W-block (r_w / z_w length nq,
// outcome-major).  Separable & fully parallel: per-outcome H_j^{-ᵀ} solve (over q),
// per-location q×q M_i^{-1} multiply (over n), per-outcome H_j^{-1} solve (over q).
void SpIOX::vadu_mv_apply(const double* r_w, double* z_w){
  arma::mat RW(const_cast<double*>(r_w), n, q, false, true);
  arma::mat T(n, q);                                   // t_j = H_j^{-ᵀ} r_j
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
  for(int j = 0; j < (int)q; ++j) T.col(j) = daggps[j].Ht_solve_A(RW.col(j));

  arma::mat Tt = T.t();                                // q×n: column i = t_i
  arma::mat St(q, n);                                  // s_i = M_i^{-1} t_i
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
  for(int i = 0; i < (int)n; ++i) St.col(i) = bw_vadu_Minv[i] * Tt.col(i);
  arma::mat S = St.t();                                // n×q

#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
  for(int j = 0; j < (int)q; ++j){                     // z_j = H_j^{-1} s_j
    arma::vec Zj = daggps[j].H_solve_A(S.col(j));
    std::copy(Zj.begin(), Zj.end(), z_w + (arma::uword)j * n);
  }
}


void SpIOX::update_Ddiag_gibbs(){
  arma::mat E = YXB - W;

  // priors for tau_sq
  double a = 2; // 1e-5;
  double b = 1; // 1e-5;
  
  // Updating each tau_sq
  for(int j=0; j<q; j++){
    arma::uvec ix = avail_by_outcome(j);
    
    double navail = .0 + avail_by_outcome(j).n_elem;

    // MCMC - sample with inverse gamma for each tau_sq
    arma::vec ej = E.col(j);
    arma::vec ej_sub = ej(ix);
    
    double ssq = arma::accu(arma::square(ej_sub));
    Ddiag(j) = 1.0/R::rgamma(navail/2 + a, 1.0/(b + 0.5 * ssq));
  
  }
}

void SpIOX::update_Ddiag_vi(){
  arma::mat E = YXB - W;
  
  // priors for tau_sq
  double a = 2;//1e-5;
  double b = 1;//1e-5;
  
  Ddiag_UQ = arma::zeros(q);
  ETE = arma::zeros(q, q);

  for(int j = 0; j < q; j++){
    arma::uvec ix = avail_by_outcome(j);
    arma::vec ej = E.col(j);
    arma::vec ej_sub = ej(ix);
    
    if(ej_sub.n_elem > 0){
      ej_sub -= arma::mean(ej_sub);
      ETE(j, j) = arma::accu(arma::square(ej_sub)); 
    }
  }
  update_running_means(ETE_ma, ETE);
  
  // Updating each tau_sq
  for(int j=0; j<q; j++){
    double navail = -1.0 + avail_by_outcome(j).n_elem;
    Ddiag_UQ(j) = ETE_ma(j,j);
    Ddiag(j) = (b + 0.5 * Ddiag_UQ(j)) / (navail/2 + a - 1);
  }
}

void SpIOX::W_centering(){
  if(intercept != -1){
    // we have an intercept. move the mean of W to it
    arma::rowvec w_means = arma::mean(W, 0);
    W.each_row() -= w_means;
    B.row(intercept) += w_means;
    YXB.each_row() -= w_means;
  }
}

void SpIOX::update_Sigma_iwishart(){
  arma::mat Smean = V.t() * V + arma::eye(V.n_cols, V.n_cols);
  arma::mat Q_mean_post;
  
  try { 
    Q_mean_post = arma::inv_sympd(Smean);
  } catch (...) {
    Rcpp::Rcout << Smean << std::endl;
    // Rcpp::Rcout << theta << std::endl; // uncomment if theta is in scope
    Rcpp::stop("Error in inv_sympd within sample_Sigma_iwishart \n");
  }
  
  double df_post = n + (V.n_cols);
  
  Q = arma::symmatu(arma::wishrnd(Q_mean_post, df_post));
  Si = arma::chol(Q, "lower");
  S = arma::inv(arma::trimatl(Si));
  Sigma = S.t() * S;
  
  // future
  //A = S.t();
  //Aplus = arma::pinv(A);
  //AplusT = Aplus.t();
}

void SpIOX::update_Sigma_vi(){
  if(latent_model > 0){
    update_running_means(VTV_ma, VTV);
    Sigma_UQ = arma::eye(q,q) + VTV_ma;
  } else {
    arma::mat HX_mat(HX.memptr(), n, p * q, false, true);   // no-copy view of the cube
    arma::mat G = HX_mat.t() * HX_mat;                       // (pq x pq)
    
    arma::mat E2(q, q, arma::fill::zeros);
    
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (int a = 0; a < (int)q; ++a) {
      for (unsigned int b = a; b < q; ++b) {                 // upper triangle only
        E2(a, b) = arma::accu(
          Beta_UQ.submat(a * p, b * p, (a + 1) * p - 1, (b + 1) * p - 1) %
            G.submat(a * p, b * p, (a + 1) * p - 1, (b + 1) * p - 1)
        );
      }
    }
    
    Sigma_UQ = arma::eye(q, q) + V.t() * V + arma::symmatu(E2);
  }
  
  double df_post = q + n;
  Sigma = Sigma_UQ / (df_post - q - 1);  
  Q = arma::symmatu(arma::inv_sympd(Sigma));
  Si = arma::chol(Q, "lower");
  S = arma::inv(arma::trimatl(Si));
  
  // future
  //A = S.t();
  //Aplus = arma::pinv(A);
  //AplusT = Aplus.t();
}


void SpIOX::sample_Y_misaligned(const arma::uvec& theta_changed){
  // precompute stuff in parallel so we can do fast sequential sampling after
  int nfill = rows_with_missing.n_elem;
  arma::mat mvnorm = arma::randn(q, nfill);
  
  // for numerical stability --
  arma::vec col_sd(q);
  arma::vec col_mean(q); // residual mean, should be ~0 but just in case
  
  for(int j = 0; j < q; j++){
    arma::uvec av = avail_by_outcome(j);
    // residuals for available observations
    arma::vec resid_j = Y(av, arma::uvec({(unsigned)j})) - 
      X.rows(av) * B.col(j);
    col_sd(j) = arma::stddev(resid_j);
    col_mean(j) = arma::mean(resid_j);
    if(col_sd(j) < 1e-10) col_sd(j) = 1.0;
  }
  // 

  arma::field<arma::mat> Hw(nfill);
  arma::field<arma::mat> Rw(nfill);
  arma::field<arma::mat> invcholP(nfill);
  
  
  if(arma::any(theta_changed != 0)){
    
    // perform this update if theta has changed and we need to recompute the
    // GP-related matrices that depend on it
    cache_blanket_comps(theta_changed);
  }
  
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
  for(int ix=0; ix<nfill; ix++){
    // this row has missing data
    int i = rows_with_missing(ix);
    Rw(ix) = Q % Rw_no_Q(ix);
    // assume all the same dag otherwise we go cray
    arma::uvec mblanket = daggps[0].mblanket(i);
    int mbsize = mblanket.n_elem;
    arma::mat Pblanket = arma::zeros(q, q*mbsize);
    
    for (int s = 0; s < q; ++s) {
      int startcol = s * mbsize;
      int endcol = (s + 1) * mbsize - 1;
      Pblanket.cols(startcol, endcol) = arma::diagmat(Q.col(s)) * Pblanket_no_Q(ix).cols(startcol, endcol);
    }
    Hw(ix) = - Pblanket;
    invcholP(ix) = arma::inv(
      arma::trimatl(arma::chol(Rw(ix), "lower")));
  }
  
  // visit every location with missing data and fill 
  // conditional on what's available and markov blanket
  for(int ix=0; ix<nfill; ix++){
    arma::vec rnormq = mvnorm.col(ix); // preloaded sample size q
    // this row has missing
    int i = rows_with_missing(ix);
    arma::uvec mblanket = daggps[0].mblanket(i);
    arma::mat YXB_others = YXB.rows(mblanket);
    
    arma::uvec which_missing = arma::find(missing_mat.row(i) == 1);
    if(which_missing.n_elem == q){
      // everything missing!
      arma::mat meancomp = invcholP(ix) * Hw(ix) * arma::vectorise( YXB_others );
      Y.row(i) = arma::trans(invcholP(ix).t() * (meancomp + rnormq)) + X.row(i) * B;
    } else {
      // some data available at this location
      arma::mat joint_cov = invcholP(ix).t() * invcholP(ix);
      arma::vec joint_mean = joint_cov * Hw(ix) * arma::vectorise( YXB_others );
      
      arma::uvec which_availab = arma::find(missing_mat.row(i) == 0);
      
      arma::mat Ckk = joint_cov(which_availab, which_availab);
      arma::mat Ckx = joint_cov(which_availab, which_missing);
      arma::mat Cxx = joint_cov(which_missing, which_missing);
      arma::mat HmatT = arma::solve(Ckk, Ckx);
      
      arma::mat cholRmat = arma::chol(arma::symmatu(Cxx - Ckx.t() * HmatT), "lower");
      
      arma::vec Yall = arma::trans(YXB.row(i));
      Yall(which_missing) = joint_mean(which_missing) + 
        HmatT.t() * (Yall(which_availab) - joint_mean(which_availab)) + 
        cholRmat * rnormq(which_missing);
      
      Y.row(i) = Yall.t() + X.row(i) * B;
    }
    
    // for numerical stability 
    for(unsigned jj = 0; jj < which_missing.n_elem; jj++){
      unsigned j = which_missing(jj);
      // fitted value for this observation and outcome
      double fitted_ij = arma::dot(X.row(i), B.col(j));
      double lo = fitted_ij + col_mean(j) - 3.0 * col_sd(j);
      double hi = fitted_ij + col_mean(j) + 3.0 * col_sd(j);
      double val = Y(i, j);
      Y(i, j) = val < lo ? lo : (val > hi ? hi : val);
    }
    
  }
  
  YXB = Y - X*B;
  
}

void SpIOX::update_BWSigma_px(){
  arma::mat SA = arma::zeros(q,q);
  arma::mat randnormmat = arma::randn(p+q, q);
  
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
  for(unsigned int j=0; j<q; j++){
    arma::uvec ix = avail_by_outcome(j);
    
    arma::vec yj = Y.col(j);
    arma::mat Zj = daggps[j].H_solve_A(V);
    
    yj = yj.rows(ix);
    arma::mat ZZ = arma::join_horiz(X.rows(ix), Zj.rows(ix));
    
    arma::mat prior_precision = arma::zeros(p+q, p+q);
    prior_precision.submat(0, 0, p-1, p-1) = arma::diagmat(1.0/B_Var.col(j));
    
    arma::mat post_precision = prior_precision + ZZ.t() * ZZ / Ddiag(j);
    arma::mat cholP = arma::chol(arma::symmatu(post_precision), "lower");
    
    arma::mat cholV = arma::inv(arma::trimatl(cholP));
    arma::vec BAj = cholV.t() * ( cholV * ZZ.t() * yj / Ddiag(j) + randnormmat.col(j));
    
    B.col(j) = BAj.head_rows(p);
    SA.col(j) = BAj.tail_rows(q);
    
    W.col(j) = Zj * SA.col(j);
    YXB.col(j) = Y.col(j) - X * B.col(j);
  }
  
  // SA^T Sigma SA is symmetric in theory; arma::symmatu cleans tiny FP
  // asymmetry that triple matrix products introduce, which would otherwise
  // trip arma::chol's symmetry check.
  Sigma = arma::symmatu(SA.t() * Sigma * SA);
  S = arma::chol(Sigma, "upper");
  Si = arma::inv(arma::trimatu(S));
  Q = Si * Si.t();
  
  // future
  //A = S.t();
  //Aplus = arma::pinv(A);
  //AplusT = Aplus.t();
}

void SpIOX::response_gibbs(int it, int sample_sigma, bool sample_beta, bool update_theta, bool sample_tausq){
  
  // update atoms for theta
  tstart = std::chrono::steady_clock::now();
  arma::uvec theta_has_changed = arma::zeros<arma::uvec>(q);
  if(update_theta){
    if(q>2){
      theta_has_changed = upd_theta_metrop_conditional(); 
    } else {
      bool block_changed = upd_theta_metrop();
      theta_has_changed += block_changed;
    }
  }
  timings(4) += time_count(tstart);
  
  if(sample_beta){
    // sample B 
    tstart = std::chrono::steady_clock::now();
    update_B();
    timings(0) += time_count(tstart);  
    
    // need to recompute V only when V = Y-XB (response model)

    tstart = std::chrono::steady_clock::now();
    compute_V();
    timings(1) += time_count(tstart);
  
  }

  
  tstart = std::chrono::steady_clock::now();
  // response model -- do we have missing data? if so, impute
  if(Y_needs_filling){
    // redo_cache_blanket runs if update_theta=true
    sample_Y_misaligned(theta_has_changed);
    compute_V();
  }
  timings(5) += time_count(tstart);

  
  if(sample_sigma > 0){
    tstart = std::chrono::steady_clock::now();
    update_Sigma_iwishart();
    timings(2) += time_count(tstart); 
  }
  
}

void SpIOX::latent_gibbs(int it, int sample_sigma, bool sample_beta, bool update_theta, bool sample_tausq){
  
  //Rcpp::Rcout << "theta " << endl;
  // update atoms for theta
  tstart = std::chrono::steady_clock::now();
  arma::uvec theta_has_changed = arma::zeros<arma::uvec>(q);
  if(update_theta){
    if(q>2){
      theta_has_changed = upd_theta_metrop_conditional();
    } else {
      bool block_changed = upd_theta_metrop();
      theta_has_changed += block_changed;
    }
  }
  timings(4) += time_count(tstart);
  
  if(latent_model == 1){
    // Block latent sampler.  Preconditioner dispatch (all precision-domain joint
    // (B,W) PCG via gibbs_BW_block, or blocked W|B if joint_BW = false):
    // PRECOND_JACOBI / VADU / POSTCOV; PRECOND_AUTO resolves to VADU.
    int cg_iter = 0;
    PrecondChoice precond_used_this_iter;

    // One full W-sampling sweep with PC `pc`, CG cap `cap` (0 = uncapped), returning
    // the W-solve CG iter count.
    //   joint_BW = true  : sample (B, W) jointly via gibbs_BW_block.
    //   joint_BW = false : blocked — B|W (conjugate update_B), then W|B precision-domain
    //                      (gibbs_w_block_precision); an ASIS non-centred B refresh is
    //                      appended after the sweep (see below).
    auto run_sweep = [&](PrecondChoice pc, int cap)->int{
      int iters = 0;
      if(joint_BW){
        gibbs_BW_block(iters, pc, /*sampling=*/true, cap);
      } else {
        if(sample_beta){
          tstart = std::chrono::steady_clock::now();
          update_B();
          timings(0) += time_count(tstart);
        }
        gibbs_w_block_precision(iters, pc, /*sampling=*/true, cap);
      }
      return iters;
    };

    // AUTO resolves to VADU; otherwise honour the explicit choice.
    precond_used_this_iter = (precond_choice == PRECOND_AUTO) ? PRECOND_VADU : precond_choice;
    cg_iter = run_sweep(precond_used_this_iter, 0);

    // ASIS interweaving (blocked route only): a non-centred B update holding
    // eta = XB + W fixed, then recover W.  Mirrors the latent_model 2/3 path and
    // breaks the B–W posterior correlation that the blocked Gibbs leaves behind.
    if(!joint_BW && sample_beta){
      int asis_iters = 0;
      tstart = std::chrono::steady_clock::now();
      update_BW_asis(asis_iters, B, W, /*sampling=*/true);
      YXB = Y - X * B;
      timings(5) += time_count(tstart);
    }

    // Surface telemetry to the outer MCMC driver.
    last_cg_iter      = cg_iter;
    last_precond_used = static_cast<int>(precond_used_this_iter);
  } else {
    // Non-block latent samplers — they manage B, W, and any inner CG themselves.
    //   latent_model == 2 : single-site Gibbs (w_sequential_singlesite, no PC dispatch).
    //   latent_model == 3 : per-outcome / single-outcome sequential sampler
    //                       (gibbs_w_sequential_byoutcome).  Honours JACOBI / VADU
    //                       directly via precond_choice; AUTO and POSTCOV resolve
    //                       to VADU (sampling=3's A_j is already per-outcome).
    if(sample_beta){
      tstart = std::chrono::steady_clock::now();
      update_B();
      timings(0) += time_count(tstart);
    }

    if(latent_model == 2){
      // redo_cache_blanket runs if update_theta=true
      w_sequential_singlesite(theta_has_changed);
    }
    if(latent_model == 3){
      // Per-outcome sequential sampler.  Honours JACOBI / VADU directly; AUTO (and
      // POSTCOV, which has no per-outcome cross-coupling) resolve to VADU.
      int cg_iter_seq = 0;
      PrecondChoice pc_used = (precond_choice == PRECOND_AUTO) ? PRECOND_VADU : precond_choice;
      gibbs_w_sequential_byoutcome(cg_iter_seq, pc_used);
      last_cg_iter      = cg_iter_seq;
      last_precond_used = static_cast<int>(pc_used);
    }
    timings(5) += time_count(tstart);

    if(sample_beta){
      int cg_iter = 0;
      update_BW_asis(cg_iter, B, W, true); // do sample
      YXB = Y - X*B;
    }
  }

  compute_V(); // keep 
  
  if(sample_sigma){
    //Rcpp::Rcout << "Sigma centered " << endl;
    tstart = std::chrono::steady_clock::now();
    update_Sigma_iwishart();
    
    //Rcpp::Rcout << "Sigma PX " << endl;
    // PX W~Sigma
    update_BWSigma_px();
    timings(2) += time_count(tstart); 
  }

  //Rcpp::Rcout << "W centering " << endl;
  W_centering(); // move to intercept if we have one
  compute_V();
  
  //Rcpp::Rcout << "Tausq " << endl;
  if(sample_tausq){
    update_Ddiag_gibbs();
  }
  
}

void SpIOX::response_vi(){
  //Rcpp::Rcout << "B\n";
  update_B();
  //Rcpp::Rcout << "V\n";
  // V = whitened Y-XB or W
  compute_V();
  //Rcpp::Rcout << "S\n";
  update_Sigma_vi();
}

void SpIOX::latent_vi(){
  auto t0 = std::chrono::high_resolution_clock::now();
  auto t_prev = t0;
  auto checkpoint = [&](const char* label){
    auto t_now = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t_now - t_prev).count();
    //Rcpp::Rcout << "[latent_vi] " << label << ": " << ms << " ms\n";
    t_prev = t_now;
  };
  
  int cg_iter = 0;
  // VI uses the joint BW block sampler.  Default PC is POSTCOV (the joint
  // block-Vecchia posterior-covariance preconditioner); VADU / JACOBI stay
  // selectable via a precond_choice override.
  const PrecondChoice use_pc =
      (precond_choice == PRECOND_VADU || precond_choice == PRECOND_JACOBI)
          ? precond_choice : PRECOND_POSTCOV;
  gibbs_BW_block(cg_iter, use_pc);
  checkpoint("gibbs_BW_block");

  // Surface CG telemetry to the VI driver (mirrors latent_gibbs) so the fit
  // loop can report CG iters / PC per printed iteration.
  last_cg_iter      = cg_iter;
  last_precond_used = static_cast<int>(use_pc);
  
  W_centering();
  checkpoint("W_centering #1");
  
  // working V with sampled B and W
  compute_V();
  checkpoint("compute_V #1");
  
  // PX
  update_BWSigma_px();
  checkpoint("update_BWSigma_px");
  
  W_centering();
  checkpoint("W_centering #2");
  
  compute_V();
  checkpoint("compute_V #2");
  
  // Update E(Sigma) using MC approx of E(quad forms)
  update_Sigma_vi();
  checkpoint("update_Sigma_vi");
  
  // Update E(Ddiag) using MC approx of E(quad forms)
  update_Ddiag_vi(); 
  checkpoint("update_Ddiag_vi");
  
  // compute covariance in B iteratively
  vi_Beta_UQ();
  checkpoint("vi_Beta_UQ");
  
  // update running means for E(B) and E(W)
  update_running_means(E_B, B);
  update_running_means(E_W, W, false); //**
  checkpoint("update_running_means");
  
  double total_ms = std::chrono::duration<double, std::milli>(
    std::chrono::high_resolution_clock::now() - t0).count();
  //Rcpp::Rcout << "[latent_vi] TOTAL: " << total_ms << " ms (cg_iter=" 
  //            << cg_iter << ")\n";
  
  vi_it ++;
}





double SpIOX::latent_fit_eval(){
  // for vi
  arma::mat E = Y - X * E_B - E_W;
  
  double ll = 0.0;
  for(int j=0; j<q; j++){
    arma::uvec ix = avail_by_outcome(j);
    
    double navail = .0 + avail_by_outcome(j).n_elem;
    
    arma::vec ej = E.col(j);
    arma::vec ej_sub = ej(ix);
    
    double ssq = arma::accu(arma::square(ej_sub));
    double dj = Ddiag(j); // diagonal element of D
    
    ll += -0.5 * navail * log(2.0 * M_PI * dj) - 0.5 * ssq / dj;
  }
  
  return ll;
}




