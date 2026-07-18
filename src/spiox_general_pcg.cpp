#include "spiox.h"
#include "sparse_solvers.h"

#include <RcppArmadillo.h>
#include <algorithm>
#include <cmath>

using namespace std;

// This file (in the code change notes) is called src/spiox_weighted.cpp

// function for doing the matrix-free PCG that is generalized to quadratic likelihood distributions
void SpIOX::general_BW_block(const arma::mat& omega, const arma::mat& Gscore,
                             const arma::mat& prior_Q, const arma::mat& prior_Q_sqrt,
                             int& cg_iter, PrecondChoice precond, bool sampling,
                             int cg_maxit_override, bool force_rebuild){
  // checking the taken in objects
  if(omega.n_rows != n || omega.n_cols != q) Rcpp::stop("omega must have dimensions n x q.");
  if(Gscore.n_rows != n || Gscore.n_cols != q) Rcpp::stop("Gscore must have dimensions n x q.");
  if(prior_Q.n_rows != q || prior_Q.n_cols != q) Rcpp::stop("prior_Q must have dimensions q x q.");
  if(prior_Q_sqrt.n_rows != q || prior_Q_sqrt.n_cols != q) Rcpp::stop("prior_Q_sqrt must have dimensions q x q.");
  
  // omega has to be non-negative
  if(omega.min() < -1e-12) Rcpp::stop("omega must be nonnegative.");
  
  arma::mat omega_safe = omega;
  omega_safe.transform([](double x){
    return x < 0.0 ? 0.0 : x;
  });
  
  const arma::uword Nb = p * q;
  const arma::uword Nw = n * q;
  
  // ----- joint operator P_joint * (b ; w) -----
  // P_joint = blkdiag(Λ_B, Λ_W) + E^T D^{-1} E,  E = (A, I), A = I_q ⊗ X
  // Block layout in the vector: head = vec(B), tail = vec(W)
  auto post_prec_mv = [&](const arma::vec& x_in, arma::vec& y_out){
    arma::mat Bin (const_cast<double*>(x_in.memptr()),       p, q, false, true);
    arma::mat Win (const_cast<double*>(x_in.memptr() + Nb),  n, q, false, true);
    arma::mat Bout(y_out.memptr(),                           p, q, false, true);
    arma::mat Wout(y_out.memptr() + Nb,                      n, q, false, true);
    
    // res = X*Bin + Win  (n × q),  res_sc = invD ⊙ res
    arma::mat res    = X * Bin + Win;
    arma::mat res_sc = omega_safe % res;
    
    // Λ_W * Win:  (Λ_W w).col(i) = H_i^T (Hw Q).col(i),  Hw.col(j) = H_j Win.col(j)
    arma::mat Hw(n, q);
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
    for(unsigned int j = 0; j < q; ++j) Hw.col(j) = daggps[j].H_times_A(Win.col(j));
    arma::mat HwQ = Hw * prior_Q;
    
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
    for(unsigned int i = 0; i < q; ++i)
      Wout.col(i) = daggps[i].Ht_times_A(HwQ.col(i)) + res_sc.col(i);
    
    // B-block:  out_B[:,j] = (1/B_Var[:,j]) ⊙ Bin[:,j] + X^T res_sc[:,j]
    arma::mat invBVar = 1.0 / B_Var;
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
    for(unsigned int j = 0; j < q; ++j)
      Bout.col(j) = invBVar.col(j) % Bin.col(j) + X.t() * res_sc.col(j);
  };
  
  // ----- preconditioner -----
  // PC-specific state lives at function scope so the lambdas (captured by
  // reference) hold valid pointers throughout the pcg_mf call.
  arma::vec Mdiag_vec;                   // JACOBI
  // PC factors live in members (bw_chol_MBj / bw_vadu_Minv / postcov factors).
  // The rebuild cadence is governed by cg_rebuild (see pc_rebuild_now): a PC only
  // accelerates CG and never shifts the target, so a frozen ("once") build is
  // still valid.  Default cadence: POSTERIOR's expensive FSAI factor is built
  // ONCE and frozen at the autostart θ/Ddiag; VADU's cheap Σ/Ddiag pieces are
  // rebuilt ALWAYS (every sweep) to track the live operator (see below).
  std::function<void(const arma::vec&, arma::vec&)> apply_Minv;
  // W-half apply (rW -> zW, each length nq).  Every non-JACOBI PC sets this; the
  // shared symmetric block Gauss-Seidel wrapper below combines it with the exact
  // B-solve so the B-W coupling (K = D^{-1}A) is preconditioned, not dropped.
  std::function<void(const double*, double*)> apply_W;
  
  if(precond == PRECOND_JACOBI){
    // Diagonal of the joint precision operator.
    //   W-half : Q(i,i) · (H_i^T H_i)(c,c) + invD_mat(c,i)
    //   B-half : 1/B_Var(c,j)            + sum_i X(i,c)^2 · invD(i,j)
    arma::mat H_col_sq(n, q);
    for(unsigned int i = 0; i < q; ++i){
      H_col_sq.col(i) = daggps[i].H_col_squared_norms();
    }
    arma::mat Mdiag_W(n, q);
    for(unsigned int i = 0; i < q; ++i)
      Mdiag_W.col(i) = prior_Q(i, i) * H_col_sq.col(i) + omega_safe.col(i);
    arma::mat Mdiag_B = (1.0 / B_Var) + arma::square(X).t() * omega_safe;
    
    Mdiag_vec.set_size(Nb + Nw);
    Mdiag_vec.head(Nb) = arma::vectorise(Mdiag_B);
    Mdiag_vec.tail(Nw) = arma::vectorise(Mdiag_W);
    
    const double diag_floor = 1e-12;
    for(arma::uword i = 0; i < Mdiag_vec.n_elem; ++i)
      if(!(Mdiag_vec(i) > diag_floor)) Mdiag_vec(i) = diag_floor;
    
    apply_Minv = [&](const arma::vec& r_in, arma::vec& z_out){
      z_out = r_in / Mdiag_vec;
    };
    
  } else if(precond == PRECOND_VADU){
    // MULTIVARIATE "Vecchia approximation with diagonal update" (Kündig & Sigrist,
    // multivariate).  Reuses the per-outcome prior Vecchia factors H_j and folds the
    // likelihood diagonal into them, coupling outcomes EXACTLY (full Q):
    //   P_VADU = Hᵀ(Q⊗I_n + diag(R⊙w))H,   H = blkdiag(H_1,…,H_q),
    // exact on the prior (Hᵀ(Q⊗I)H IS the IOX prior precision) and folding the
    // likelihood per outcome as H_jᵀ diag(R_j⊙w_j) H_j = B_jᵀ diag(w_j) B_j ≈ diag(w_j).
    // The middle M = Q⊗I_n + diag(R⊙w) is block-diagonal in LOCATION order with
    // M_i = Q + diag_j(R_i^{(j)}w_i^{(j)}), so P^{-1} = H^{-1}M^{-1}H^{-ᵀ} applies as a
    // SEPARABLE, fully-parallel sweep: per-outcome H_j^{-ᵀ}/H_j^{-1} solves around a
    // per-location q×q M_i^{-1} (build_vadu_Minv / vadu_mv_apply).  Reduces to the
    // scalar dscale VADU when Q is diagonal.  B half: exact per-outcome dense Cholesky.
    
    // VADU's only θ-dependent (expensive) object is the prior Vecchia factor H_j (via
    // daggps[j]), which the θ-update owns — the PC never rebuilds it.  Every
    // Σ/Ddiag-dependent piece (M_i^{-1}, MBj) is O(n·q³) cheap, so the default cadence
    // (cg_rebuild = "always") RECOMPUTES both from the live Σ/Ddiag every sweep, keeping
    // the PC tracking the current operator instead of going stale (which makes a frozen
    // VADU's CG count climb as Σ/Ddiag drift).  cg_rebuild = "once" freezes them at the
    // autostart values instead (guarded by vadu_pc_n_builds).
    
    // For now, we force it to rebuild every iter, bcs Q and omega are updated every iter
    if(force_rebuild || pc_rebuild_now(PRECOND_VADU, vadu_pc_n_builds)){
      auto t_pc = std::chrono::steady_clock::now();
      // B half: per-outcome p×p Cholesky from the current invD.
      bw_chol_MBj.assign(q, arma::mat());
      for(unsigned int j = 0; j < q; ++j){
        arma::mat DX = X;
        DX.each_col() %= omega_safe.col(j);
        arma::mat MBj = X.t() * DX;
        MBj.diag()  += 1.0 / B_Var.col(j);
        bw_chol_MBj[j] = arma::chol(arma::symmatu(MBj), "upper");
      }
      // Multivariate middle: per-location M_i^{-1} = (Q + diag_j R_i^{(j)}w_i^{(j)})^{-1}.
      build_vadu_Minv_general(omega_safe, prior_Q);
      ++vadu_pc_n_builds;
      pc_build_seconds += time_count(t_pc) / 1e6;
    }
    
    // W half: P_VADU^{-1} = H^{-1} M^{-1} H^{-ᵀ} — per-outcome triangular solves
    // around the per-location q×q M_i^{-1} (exact cross-outcome Q coupling).
    apply_W = [&](const double* rW, double* zW){ vadu_mv_apply(rW, zW); };
    
  } else if(precond == PRECOND_POSTCOV){
    // Multivariate latent posterior-conditional PC.  W half: a SINGLE block-Vecchia
    // factor of the joint posterior covariance (q×q blocks couple all outcomes per
    // location); no R_corr mix (the coupling lives in the blocks).  Decouples per
    // outcome at diagonal Σ; cadence per cg_rebuild (AUTO → ALWAYS).  (B half +
    // B–W coupling handled by the shared block Gauss-Seidel wrapper after the chain.)
    if(force_rebuild || pc_rebuild_now(PRECOND_POSTCOV, postcov_n_builds)){
      auto t_pc = std::chrono::steady_clock::now();
      bw_chol_MBj.assign(q, arma::mat());
      for(unsigned int j = 0; j < q; ++j){
        arma::mat DX = X;
        DX.each_col() %= omega_safe.col(j);
        arma::mat MBj = X.t() * DX;
        MBj.diag()  += 1.0 / B_Var.col(j);
        bw_chol_MBj[j] = arma::chol(arma::symmatu(MBj), "upper");
      }
      build_postcov_factors_general(omega_safe, prior_Q);
      pc_build_seconds += time_count(t_pc) / 1e6;
    }
    // W half: block fwd/back substitution over the joint factor.
    apply_W = [&](const double* rW, double* zW){ postcov_apply(rW, zW); };
    
  }
  
  // Combine the exact B-solve with the chosen W-half (apply_W) via SYMMETRIC BLOCK
  // GAUSS-SEIDEL on the joint precision P = [[M_BB, Kᵀ],[K, A_W]], K = D^{-1}A (the B–W
  // coupling the plain block-diagonal PC dropped — strong under fixed-effect/spatial
  // confounding):
  //   u_B = M_BB^{-1} r_B
  //   z_W = A_W^PC ( r_W − D^{-1}X·u_B )
  //   z_B = M_BB^{-1} ( r_B − XᵀD^{-1}·z_W )
  // SPD whenever M_BB and A_W^PC are SPD, so it stays a valid CG preconditioner; costs
  // one extra exact p×p B-solve + two cheap coupling mults over the block-diagonal PC.
  // JACOBI keeps its own (pure-diagonal) apply_Minv; every other PC is wrapped here.
  if(precond != PRECOND_JACOBI){
    apply_Minv = [&](const arma::vec& r_in, arma::vec& z_out){
      arma::mat RB(const_cast<double*>(r_in.memptr()),      p, q, false, true);
      arma::mat RW(const_cast<double*>(r_in.memptr() + Nb), n, q, false, true);
      arma::mat ZB(z_out.memptr(),                          p, q, false, true);
      arma::mat ZW(z_out.memptr() + Nb,                     n, q, false, true);
      auto Bsolve = [&](const arma::mat& rhs, arma::mat& out){
        for(unsigned int j = 0; j < q; ++j){
          arma::vec tmp = arma::solve(arma::trimatl(bw_chol_MBj[j].t()), rhs.col(j),
                                      arma::solve_opts::fast);
          out.col(j)    = arma::solve(arma::trimatu(bw_chol_MBj[j]),     tmp,
                  arma::solve_opts::fast);
        }
      };
      arma::mat uB(p, q);
      Bsolve(RB, uB);                                              // u_B = M_BB^{-1} r_B
      arma::vec tWv = arma::vectorise(RW - omega_safe % (X * uB));   // r_W − K u_B
      apply_W(tWv.memptr(), z_out.memptr() + Nb);                  // z_W = A_W^PC(·) -> ZW
      arma::mat tB = RB - X.t() * (omega_safe % ZW);                 // r_B − Kᵀ z_W
      Bsolve(tB, ZB);                                              // z_B = M_BB^{-1} t_B
    };
  }
  
  arma::mat cW = Gscore;
  arma::mat cB = X.t() * Gscore;
  
  // prior noise on W (same Unorm trick as gibbs_w_block)
  arma::mat Unorm, xi_B_prior, Zlik_sc;
  if(sampling){
    Unorm = arma::randn(n, q) * prior_Q_sqrt.t();
    for(unsigned int j = 0; j < q; ++j) Unorm.col(j) = daggps[j].Ht_times_A(Unorm.col(j));
    xi_B_prior = arma::randn(p, q) / arma::sqrt(B_Var);
    Zlik_sc = arma::randn(n, q) % arma::sqrt(omega_safe);
  } else {
    Unorm = arma::zeros(n, q);
    xi_B_prior = arma::zeros(p, q);
    Zlik_sc = arma::zeros(n, q);
  }
  
  // likelihood noise — SAME draw shared between B and W parts to match Λ_lik = E^T D^{-1} E
  arma::mat xi_B_lik = X.t() * Zlik_sc;
  
  arma::mat RHS_B = cB + xi_B_prior + xi_B_lik;
  arma::mat RHS_W = cW + Unorm     + Zlik_sc;
  
  arma::vec rhs(Nb + Nw);
  rhs.head(Nb) = arma::vectorise(RHS_B);
  rhs.tail(Nw) = arma::vectorise(RHS_W);
  
  // Cold-start CG (x0 = 0).  Warm-starting from the previous (B, W) makes
  // the initial residual r_0 already small in M-norm whenever the chain has
  // settled and the preconditioner is close to exact (PPCG with prior-
  // dominated posterior is exactly this regime).  CG then satisfies the
  // M-norm relative-residual stopping criterion in a single step — but a
  // 1-step Bhattacharya sample is biased because alpha_0 ≠ 1 systematically
  // shrinks the move from x_0 toward M^{-1} b.  Starting from 0 forces CG
  // to do enough Krylov iterations to faithfully transport the noise
  // contribution of the RHS into the sample, removing the bias.
  arma::vec x0 = arma::zeros<arma::vec>(Nb + Nw);
  
  // maxit defaults to n; the probe uses cg_maxit_override to cap JACOBI's
  // budget at 2·max(POSTERIOR iters) for a fair head-to-head comparison.
  const int cg_maxit = (cg_maxit_override > 0)
    ? cg_maxit_override
  : static_cast<int>(npq);
  arma::vec sol = pcg_mf(post_prec_mv, apply_Minv, cg_iter, rhs, x0,
                         5*1e-5, cg_maxit, num_threads);
  
  // unpack (B, W) but delete YXB (bcs it is for Gaussian)
  arma::mat B_old = B;
  B = arma::mat(sol.memptr(),       p, q);
  W = arma::mat(sol.memptr() + Nb,  n, q);
  //YXB += X * (B_old - B);
}

// function to build VADU preconditioners for quadratic likelihood distributions
void SpIOX::build_vadu_Minv_general(const arma::mat& omega, const arma::mat& prior_Q){
  // implementation code here
  bw_vadu_Minv.assign(n, arma::mat());
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
  for(int i = 0; i < (int)n; ++i){
    arma::mat Mi = prior_Q;
    for(int j = 0; j < (int)q; ++j){
      const double Rij = daggps[j].sqrtR(i) * daggps[j].sqrtR(i);     // R_i^{(j)} 
      Mi(j, j) += Rij * omega(i, j);
    }
    bw_vadu_Minv[i] = arma::inv_sympd(arma::symmatu(Mi));
  }
}

// function to build postcov preconditioners for quadratic likelihood distributions
void SpIOX::build_postcov_factors_general(const arma::mat& omega, const arma::mat& prior_Q){
  // implementation code here
  if(!postcov_setup_done) postcov_setup();
  
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
  for(int i = 0; i < (int)n; ++i){
    arma::vec sd(q), inv_sd(q);
    for(int j = 0; j < (int)q; ++j){ sd(j) = daggps[j].sqrtR(i); inv_sd(j) = 1.0 / sd(j); }
    arma::mat Rinv = prior_Q;                                  // R_i^{-1} = Λ^{-1} Q Λ^{-1}
    Rinv.each_col() %= inv_sd;
    Rinv.each_row() %= inv_sd.t();
    arma::mat K = Rinv;                                  // K_i = R_i^{-1} + Ω_i
    for(int j = 0; j < (int)q; ++j){ K(j, j) += omega(i, j);} 
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