#include "omp_import.h"
#include "sparse_solvers.h"
#include "daggp.h"
#include "ramadapt.h"

#include <memory>
#include <functional>

using namespace std;

int time_count(std::chrono::steady_clock::time_point tstart);

class SpIOX {
public:
  // Y = XB + Z where Z is multivariate q,q IOC-GP
  // S^T*S = Sigma = Q^-1 = (Lambda*Lambda^T + Delta)^-1. Lambda, Q are sparse
  
  // -------------- data
  
  // matrix of outcomes dim n, q 
  arma::mat Y;
  // matrix of predictors dim n, p
  arma::mat X;
  
  // metadata
  unsigned int n, q, p, nq, pq, npq;
  int intercept; // location of intercept in X, if at all
  int num_threads;
  //double spatial_sparsity;
  
  // -------------- model parameters
  // objects that depend on B
  arma::mat B;
  arma::mat YXB;
  arma::mat B_Var; // prior variance on B, element by element
  
  // objects that depend on Sigma
  arma::mat S, Si, Sigma, Q; // S^T * S = Sigma = Q^-1 = (Lambda*Lambda^T + Delta)^-1 = (Si * Si^T)^-1
  
  // future:
  //arma::mat A, Aplus, AplusT; 
  
  // objects that depend on theta RadGP for spatial dependence
  std::vector<DagGP> daggps, daggps_alt;
  arma::mat theta; // each column is one alternative value for theta
  bool gridded;
  
  // objects that depend on W
  arma::mat V; 
  
  // -------------- utilities
  int matern;
  void update_B(); // 
  void update_Sigma_iwishart();
  
  void compute_V(); 
  bool upd_theta_metrop();
  arma::uvec upd_theta_metrop_conditional(); // returns uvec with changes to thetaj
  void init_theta_adapt();
  
  // data with with misalignment
  arma::field<arma::uvec> avail_by_outcome;
  arma::umat missing_mat;
  arma::uvec rows_with_missing;
  arma::uvec rows_some;
  int n_some; // n where at least one is nonmissing
  bool Y_needs_filling;
  arma::uvec Y_na_indices;
  void manage_missing_data(){
    // managing misalignment, i.e. not all outcomes observed at all locations
    // indices of non-NAs 
    avail_by_outcome = arma::field<arma::uvec>(q);
    missing_mat = arma::zeros<arma::umat>(n, q);
    arma::uvec row_miss_01 = arma::zeros<arma::uvec>(n);
    arma::uvec row_complete_miss_01 = arma::zeros<arma::uvec>(n);
    Y_na_indices = arma::find_nonfinite(Y);
    for(int j=0; j<q; j++){
      avail_by_outcome(j) = arma::find_finite(Y.col(j));
      for(int i=0; i<n; i++){
        missing_mat(i,j) = !arma::is_finite(Y(i,j));
      }
    }
    Y.elem(arma::find(missing_mat)).zeros(); // set missing to zero
    
    for(int i=0; i<n; i++){
      row_miss_01(i) = arma::any(missing_mat.row(i) == 1);
      row_complete_miss_01(i) = arma::all(missing_mat.row(i) == 1); 
    }
    rows_with_missing = arma::find(row_miss_01 == 1);
    rows_some = arma::find(row_complete_miss_01 == 0);
    n_some = rows_some.n_elem;
    if(n_some < n){
      if(latent_model != 1){
        Rcpp::stop("Fully missing rows in Y detected. Use latent_model=1 or exclude from data and rerun.\n");
      }
    }
    //if(Y.has_nonfinite() & (latent_model==1)){
    //  Rcpp::stop("nq block not implemented for misaligned data.\n");
    //}
    if(arma::accu(missing_mat)>0){
      Y_needs_filling = true;
    } else {
      Y_needs_filling = false;
    }
  }
  void sample_Y_misaligned(const arma::uvec& theta_changed);
  
  // Preconditioner / W-sampler choices for the block latent model.
  // AUTO (the default) resolves to VADU.
  //
  //   JACOBI    : diagonal of the joint operator (precision form).
  //               O(nq + pq) per apply; nearly free; tends to be a weak PC.
  //
  //   VADU      : MULTIVARIATE "Vecchia approximation with diagonal update" (Kündig
  //               & Sigrist).  Reuses the per-outcome prior Vecchia factors H_j and
  //               folds the likelihood, coupling outcomes EXACTLY via Q:
  //                 P_VADU = Hᵀ(Q⊗I_n + diag(R⊙w))H,  H = blkdiag(H_1,…,H_q).
  //               The middle is block-diagonal by location (M_i = Q + diag_j R_i^{(j)}w_i^{(j)}),
  //               so Apply P^{-1} = H^{-1}M^{-1}H^{-ᵀ} is a SEPARABLE fully-parallel sweep:
  //               per-outcome prior triangular solves around a per-location q×q M_i^{-1}.
  //               Reduces to the scalar 1/(R_j⊙w_j+Q_jj) at Σ diagonal.  Only per-sweep
  //               build is the n cheap q×q inverses.
  //
  //   POSTCOV: ONE block-Vecchia factor of the JOINT posterior covariance (q×q blocks
  //               couple all outcomes per location); fewest CG iters on confounded /
  //               high-correlation problems, ~2× costlier per iter than VADU.  See the
  //               detailed block further down.  sampling=1 only (sampling=3 → VADU).
  enum PrecondChoice {
    PRECOND_AUTO       = 0,   // resolves to VADU (MCMC); POSTCOV in the latent VI fit
    PRECOND_JACOBI     = 1,
    PRECOND_VADU       = 2,
    PRECOND_POSTCOV    = 3
  };
  PrecondChoice precond_choice = PRECOND_AUTO;

  // How often the Σ/Ddiag-dependent preconditioner factors are rebuilt during
  // MCMC.  A preconditioner only accelerates CG and never shifts the sampled
  // target, so freezing it at the autostart values ("once") is exact, while
  // rebuilding every sweep ("always") keeps it tracking the drifting operator at
  // extra cost.
  //   REBUILD_AUTO   : per-PC default — ALWAYS for POSTCOV (cheap in-place
  //                    rebuild of the q×q blocks).
  //   REBUILD_ALWAYS : rebuild every sweep regardless of PC.
  //   REBUILD_ONCE   : build once at the autostart values, then freeze.
  // VADU is a special case: its factor always rebuilds (see pc_rebuild_now)
  // regardless of this setting, because the recompute is trivially cheap.
  enum RebuildMode { REBUILD_AUTO = 0, REBUILD_ALWAYS = 1, REBUILD_ONCE = 2 };
  int cg_rebuild = REBUILD_AUTO;
  // Resolve cg_rebuild for PC `pc` given how many times its factors have already
  // been built (`builds_done`): returns whether to (re)build this sweep.
  bool pc_rebuild_now(PrecondChoice pc, int builds_done) const {
    // VADU always rebuilds: its factor is an O(nq) trivial recompute (dscale /
    // R_corr), so freezing it saves nothing and a stale dscale only hurts CG.
    // This overrides cg_rebuild = "once" (now the global default) for VADU only.
    if(pc == PRECOND_VADU) return true;
    int mode = cg_rebuild;
    // POSTCOV defaults to ALWAYS: its rebuild is cheap (overwrites preallocated
    // q×q block values in place, parallel over locations) and tracking the live
    // Σ/Ddiag empirically beats freezing at the autostart values.
    if(mode == REBUILD_AUTO)
      mode = (pc == PRECOND_POSTCOV) ? REBUILD_ALWAYS : REBUILD_ONCE;
    return (mode == REBUILD_ALWAYS) ? true : (builds_done == 0);
  }

  // Per-location state for the MULTIVARIATE POSTCOV PC ("postcov").  Instead
  // of q separate per-outcome factors mixed by a uniform R_corr, this builds a
  // SINGLE block-Vecchia inverse-Cholesky factor U of the JOINT posterior
  // covariance, with q×q blocks coupling all outcomes at a location.  The IOX
  // prior has V = HW with rows v_i = (V_{i1..iq}) iid N(0,Σ), so the multivariate
  // prior conditional is
  //     w_i | w_{N_i} ~ N(M_i w_{N_i}, R_i),  R_i = Λ_i Σ Λ_i,
  // with Λ_i = diag_j sqrtR_i^{(j)} (the per-outcome conditional sd, from
  // daggps[j].sqrtR(i)) and M_i's parent-t block = diag_j b_i^{(j)}(t) (from
  // daggps[j].h(i)(t); the shared parents are daggps[0].dag(i)).  Folding the
  // local datum y_i (precision Ω_i = diag_j invD_ij, 0 at missing) gives the
  // posterior conditional covariance F_i = (R_i^{-1}+Ω_i)^{-1} = L_i L_iᵀ and the
  // mean coefficient G_i = F_i R_i^{-1} M_i.  The block factor U (location-major,
  // lower-block-triangular) has diagonal block L_i^{-1} and parent-t block
  // -L_i^{-1} G_i^{(t)}, so UᵀU ≈ joint posterior precision and M^{-1} = U^{-1}U^{-ᵀ}.
  // KEY (Lever 1): the parent-t block is NOT a general q×q matrix — it factors as
  //     -L_i^{-1} G_i^{(t)} = -E_i D_i^{(t)},   E_i = L_iᵀ Λ_i^{-1} Q  (LOCATION-only, q×q),
  //                                             D_i^{(t)} = diag(b_i^{(t)}/sqrtR_i)  (DIAGONAL).
  // All m parent-blocks at i share ONE E_i and differ only by a q-vector of diagonal scalings,
  // so the apply gathers parents/children with a cheap diagonal weighting (O(mq)) and does ONE
  // q×q gemv per location instead of m — per-edge work q²→q, matching VADU's flop order.  Stored
  // as dense blocks (NOT Eigen sparse — BLAS gemv beats a scalar sparse triangular solve, ~2–3×):
  //     postcov_L[i] = L_i                          (q×q lower)
  //     postcov_E[i] = L_iᵀ Λ_i^{-1} Q              (q×q;     zeros if root)
  //     postcov_D[i] = [ d_i^{(1)} | ... ]          (q × m_i; col t = b_i^{(t)}/sqrtR_i; empty if root)
  // At diagonal Σ, E_i and D_i^{(t)} are diagonal, so the block solve decouples into q
  // independent per-outcome solves.  Per-rebuild work O(n(q³+m q)) (rows independent →
  // OpenMP-parallel), reusing R_i^{-1}=Λ_i^{-1}QΛ_i^{-1} (Q=Σ^{-1} a member, one q×q SPD
  // inverse per location).
  //
  // Apply M^{-1}=U^{-1}U^{-ᵀ} is a block triangular solve — inherently serial along
  // the DAG.  We parallelise it by LEVEL SCHEDULING: level[i] = 1+max level over
  // parents; locations within a level are mutually independent, so each level is an
  // OpenMP-parallel sweep (forward = levels up, gathering from parents; back = levels
  // down, gathering from children).  Runs on two q×n buffers (the solution s and the
  // back-pass g_i = E_iᵀ s_i that i's parents read).
  // The level/children structure is DAG-only (θ/Σ/Ddiag-independent) → precomputed
  // once.  Location- (n-) parallelism (vs VADU's per-outcome q-parallelism), where
  // postcov's far fewer CG iters on confounded/high-correlation problems pays off.
  // NB: these stay DOUBLE.  fp32 storage of L_i/E_i was tried and REJECTED — at high
  // cross-outcome correlation Q=Σ⁻¹ is ill-conditioned, so R_i⁻¹=Λ⁻¹QΛ⁻¹ and hence the
  // L_i/E_i blocks have large dynamic range; fp32 loses the coupling that makes postcov
  // work and CG iters BLOW UP (45.8→132 at q=30, corr~0.99) — net much slower despite the
  // cheaper per-iter apply.  The apply is not bandwidth-bound enough to justify it here.
  int  postcov_n_builds = 0;
  bool postcov_setup_done = false;             // levels/children/storage precomputed?
  std::vector<arma::mat> postcov_L;            // size n, each q×q lower-triangular
  std::vector<arma::mat> postcov_E;            // size n, each q×q (E_i = L_iᵀ Λ_i⁻¹ Q; zeros if root)
  std::vector<arma::mat> postcov_D;            // size n, each q×m_i (col t = d_i^{(t)}; empty if root)
  arma::uvec             postcov_order;        // locations sorted by level (stable)
  arma::uvec             postcov_level_ptr;    // size n_levels+1: level slices into order
  arma::field<arma::uvec> postcov_child_k;     // per i: children k (i is a parent of k)
  arma::field<arma::uvec> postcov_child_t;     // per i: position of i in k's parent list
  arma::mat              postcov_buf;          // reused q×n location-major apply buffer (sol s)
  arma::mat              postcov_g;            // reused q×n back-pass buffer (g_i = E_iᵀ s_i)
  std::vector<arma::vec> postcov_acc;          // per-thread scratch, length q
  std::vector<arma::vec> postcov_u;            // per-thread scratch, length q (forward parent gather)
  void postcov_setup();                        // one-time levels + children + storage
  void build_postcov_factors();
  
  //Generalized version - prior_Q is the Sigma prior inverse
  void build_postcov_factors_general(const arma::mat& omega, const arma::mat& prior_Q);
  
  // Apply the postcov W-half PC to one W-block: z = U^{-1}U^{-ᵀ} r, r_w / z_w
  // length nq (outcome-major).  Level-scheduled parallel block substitution.
  void postcov_apply(const double* r_w, double* z_w);

  // MULTIVARIATE VADU.  Folds the likelihood diagonal into the per-outcome prior
  // Vecchia factors H_j and couples outcomes EXACTLY (full Q, not the old separable
  // R_corr): P_VADU = Hᵀ(Q⊗I_n + diag(R⊙w))H, H = blkdiag(H_1,…,H_q).  The middle
  // M = Q⊗I_n + diag(R⊙w) is block-diagonal in LOCATION order, M_i = Q + diag_j(R_i^{(j)}w_i^{(j)})
  // (R_i^{(j)} = sqrtR_i^{(j)2}, w_i^{(j)} = invD_ij), so the apply is SEPARABLE and fully
  // parallel — per-outcome H_j^{-ᵀ}/H_j^{-1} triangular solves (over q) around a per-location
  // q×q solve M_i^{-1} (over n); no serial DAG substitution.  Reduces to the scalar
  // dscale VADU when Q is diagonal.  bw_vadu_Minv[i] = M_i^{-1}, built per sweep.
  std::vector<arma::mat> bw_vadu_Minv;            // size n, each q×q (= M_i^{-1})
  void build_vadu_Minv();                          // per-location M_i^{-1} (parallel over n)
  
  // Generalized version for non-Gaussian dist.
  // omega is n x q and prior_Q is q x q.
  void build_vadu_Minv_general(const arma::mat& omega, const arma::mat& prior_Q);
  
  void vadu_mv_apply(const double* r_w, double* z_w);  // H^{-1} M^{-1} H^{-ᵀ}, length nq

  // Telemetry: number of CG iterations used in the most-recent W-block update,
  // plus an integer code for which preconditioner / sampler ran
  // (0 unset / 1 jacobi / 2 vadu / 3 postcov).  Read by the
  // outer MCMC driver and surfaced back to R.
  int last_cg_iter      = 0;
  int last_precond_used = 0;
  // Wall-clock seconds spent on the one-time ("once" mode) preconditioner
  // factor build (BW PC factors).  Accumulated the
  // first time the PC is constructed; 0 thereafter.  Surfaced back to R.
  double pc_build_seconds = 0.0;
  
  // latent model
  int latent_model; // 0: response, 1: block, 2: row seq, 3: col seq
  // For the block latent model (latent_model = 1):
  //   joint_BW = true  (default): sample (B, W) jointly via gibbs_BW_block.
  //   joint_BW = false          : blocked route — B|W (conjugate update_B),
  //                               then W|B in the precision domain
  //                               (gibbs_w_block_precision), followed by an ASIS
  //                               non-centred B refresh (update_BW_asis), mirroring
  //                               the latent_model 2/3 samplers.  Honours the same
  //                               JACOBI / VADU / POSTCOV choices.
  bool joint_BW = true;
  arma::mat W;
  void w_sequential_singlesite(const arma::uvec& theta_changed);

  // Per-outcome sequential sampler for W (latent_model = 3).  Uses a fixed
  // sparse-precision-form CG with Jacobi PC (pcg_diag_solve) — the precond
  // enum is accepted for ABI compatibility but ignored.
  void gibbs_w_sequential_byoutcome(int& cg_iter, PrecondChoice precond,
                                    int cg_maxit_override = 0);

  // β-conditional W update (ASIS reparameterisation) used by the non-block
  // latent samplers (latent_model = 2, 3) when sample_Beta is on.  Carries
  // its own inline Jacobi PC — doesn't touch the PrecondChoice enum.
  void update_BW_asis(int& cg_iter, arma::mat& B, arma::mat& W, bool sampling);

  // Generalized joint Gaussian update for a quadratic likelihood:
  //
  //   -0.5 * vec(XB + W)' diag(vec(omega)) vec(XB + W) + vec(Gscore)' vec(XB + W)
  // Meaning for Gaussian, Binomial, NegBin
  // prior_Q_sqrt is the lower Chol of prior_Q 
  // omega and Gscore are passed explicitly and are not stored in SpIOX.
  void general_BW_block(const arma::mat& omega, const arma::mat& Gscore,
                        const arma::mat& prior_Q, const arma::mat& prior_Q_sqrt,
                        int& cg_iter, PrecondChoice precond, bool sampling = true,
                        int cg_maxit_override = 0,
                        bool force_rebuild = false);
  
  // Joint BW PCG sampler — the only block sampler that survives the cleanup.
  // This PCG sampler is limited to Gaussian though.
  // PC dispatched on `precond`: PRECOND_JACOBI (diagonal of the joint precision),
  // PRECOND_VADU, or PRECOND_POSTCOV (each defines a W-half apply that the shared
  // symmetric block Gauss-Seidel wrapper combines with the exact p×p B-solve).
  // `sampling` toggles between MCMC sampling mode (RHS includes Bhattacharya noise
  // terms) and posterior-mean mode (deterministic).  `cg_maxit_override` > 0 caps the
  // PCG iter count; 0 means use the default cap (n).
  void gibbs_BW_block(int& cg_iter, PrecondChoice precond, bool sampling=true,
                      int cg_maxit_override=0);

  // W | B precision-domain block sampler (the W-only counterpart of
  // gibbs_BW_block, used by the blocked route when joint_BW = false).  Holds B
  // fixed (sampled separately by update_B + ASIS), solving the conditional
  // W precision P_WW = Λ_W + diag(invD) via matrix-free PCG.  PC dispatched on
  // `precond`: PRECOND_JACOBI / PRECOND_VADU / PRECOND_POSTCOV — the same W-half
  // preconditioners as gibbs_BW_block, with no B half.  `sampling` toggles
  // Bhattacharya noise on the RHS; `cg_maxit_override` > 0 caps the PCG iters,
  // 0 = default cap (n).
  void gibbs_w_block_precision(int& cg_iter, PrecondChoice precond,
                               bool sampling=true, int cg_maxit_override=0);

  // Preconditioner buffers (built per the cg_rebuild cadence).
  // chol_MBj : per-outcome p×p Cholesky of the B-half precision (shared by all
  //            non-JACOBI PCs via the block Gauss-Seidel wrapper).
  // vadu_dscale: per-outcome sqrt(R_j⊙invD_j + Q_jj) diagonal scale (VADU only).
  // vadu_pc_n_builds tracks how many times the VADU factors have been (re)built so
  // cg_rebuild = "once" can freeze them.
  int vadu_pc_n_builds = 0;
  std::vector<arma::mat> bw_chol_MBj;
  std::vector<arma::vec> bw_vadu_dscale;

  arma::vec Ddiag;
  
  // centering of W and move to intercept (if there is one)
  void W_centering();
  // update running means
  void update_running_means(arma::mat&, const arma::mat&, bool pr=false); 
  
  // utilities for gibbs
  void update_BWSigma_px();
  void update_Sigma_gibbs();
  void update_Ddiag_gibbs();
  
  // whitened Y and X (that is, applying the same operation that makes W white noise)
  arma::mat Ytilde;
  arma::cube HX;
  
  // utilities for vi
  int vi_min_iter; // burn-in for vi in latent models
  int vi_it; // internal iteration counter
  //void update_B_vi();
  void update_Sigma_vi();
  void update_Ddiag_vi();
  bool vi;
  //arma::mat B_post_cov;
  arma::mat VTV, VTV_ma; // ma for moving average
  arma::mat ETE, ETE_ma;  
  arma::mat E_B; 
  arma::mat W_RB;
  arma::mat E_W;
  
  void vi_Beta_UQ(); // for computing Beta_UQ
  arma::vec delta_t, beta_running_mean;
  arma::mat Beta_UQ;
  arma::vec Ddiag_UQ;
  arma::mat Sigma_UQ;
  
  
  // utility for latent model and misaligned response model
  arma::field<arma::mat> Rw_no_Q;
  arma::field<arma::mat> Pblanket_no_Q;
  void cache_blanket_comps(const arma::uvec& theta_changed);
  
  // which theta updates are we doing
  bool phi_sampling, sigmasq_sampling, nu_sampling, alpha_sampling;
  
  // adaptive metropolis to update theta atoms
  int theta_mcmc_counter;
  arma::uvec which_theta_elem;
  arma::mat theta_unif_bounds;
  //arma::mat theta_metrop_sd;
  RAMAdapt theta_adapt;
  bool theta_adapt_active;
  // adaptive metropolis (conditional update) to update theta atoms
  // assume shared covariance functions and unknown parameters across variables
  arma::mat c_theta_unif_bounds;
  std::vector<RAMAdapt> c_theta_adapt;
  // --------
  
  
  // -------------- run 1 gibbs iteration based on current values
  void response_gibbs(int it, int sample_sigma, bool sample_beta, bool update_theta, bool sample_tausq=false);
  void latent_gibbs(int it, int sample_sigma, bool sample_beta, bool update_theta, bool sample_tausq=false);
  void response_vi();
  void latent_vi();
  double latent_fit_eval();
  
  std::chrono::steady_clock::time_point tstart;
  arma::vec timings;
  
  // -------------- constructor for building C (dense! beware)
  SpIOX(const arma::mat& _coords,
        const arma::field<arma::uvec>& custom_dag,
        int dag_opts,
        const arma::mat& daggp_theta, 
        const arma::mat& Sigma,
        int cov_model_matern,
        int num_threads_in){
    
    theta = daggp_theta;
    
    q = theta.n_cols;
    n = _coords.n_rows;
    
    daggps = std::vector<DagGP>(q);

    matern = cov_model_matern;

    // make n_threads depend on whether data are gridded, since behavior is opposite
    gridded = dag_opts==-1;
    int daggp_n_threads = gridded ? 1 : num_threads;
    for(unsigned int i=0; i<q; i++){
      daggps[i] = DagGP(_coords, theta.col(i), custom_dag,
                        dag_opts,
                        matern,
                        daggp_n_threads);
    }
  }
  
  // -------------- constructor
  SpIOX(const arma::mat& _Y,
        const arma::mat& _X,
        const arma::mat& _coords,
        const arma::field<arma::uvec>& custom_dag,
        int dag_opts,
        int latent_model_choice,
        const arma::mat& Beta_start,
        const arma::mat& W_start,
        const arma::mat& Sigma_start,
        const arma::mat& daggp_theta,
        const arma::uvec& update_theta_which,
        const arma::vec& tausq_start,
        int cov_model_matern,
        int num_threads_in,
        int _vi_min_iter = 100)
  {
    num_threads = num_threads_in;

    Y = _Y;
    X = _X;

    n = Y.n_rows;
    q = Y.n_cols;
    p = X.n_cols;
    
    nq = n*q;
    pq = p*q;
    npq = pq + nq;
    
    // intercept? 
    intercept = -1;
    for(int j=0; j<p; j++){
      if(arma::all(X.col(j) == 1.0)){
        intercept = j;
        break;
      }
    }
    
    latent_model = latent_model_choice; // latent model?

    manage_missing_data();
    
    if(latent_model>0){
      W = W_start;
      W(arma::find_nonfinite(W)).fill(0);
      E_W = W;
      W_RB = W;
      Ddiag = tausq_start;
    }
    
    
    // mcvi params
    vi = _vi_min_iter > 0;
    vi_min_iter = _vi_min_iter;
    vi_it = 0;
    
    VTV = arma::zeros(q, q);
    ETE = arma::zeros(q, q);
    delta_t = arma::zeros(p*q);
    beta_running_mean = arma::zeros(p*q);
    Beta_UQ = arma::zeros(p*q, p*q);
    
    VTV_ma = arma::zeros(q, q);
    ETE_ma = arma::zeros(q, q);
    
    Ytilde = arma::zeros(n, q);
    HX = arma::zeros(n, p, q);
    
    // continue with other init
    B = Beta_start;
    E_B = B;
    YXB = Y - X * B;
    
    B_Var = 1000 * arma::ones(arma::size(B));
    
    theta = daggp_theta;
    daggps = std::vector<DagGP>(q);

    // if multiple nu options, interpret as wanting to sample smoothness for matern
    // otherwise, power exponential with fixed exponent.
    phi_sampling = update_theta_which(0) == 1;
    sigmasq_sampling = update_theta_which(1) == 1;
    nu_sampling = update_theta_which(2) == 1;
    alpha_sampling = update_theta_which(3) == 1;

    matern = cov_model_matern;

    // make n_threads depend on whether data are gridded, since behavior is opposite
    gridded = dag_opts==-1;
    int daggp_n_threads = gridded ? 1 : num_threads;
    for(unsigned int i=0; i<q; i++){
      daggps[i] = DagGP(_coords, theta.col(i), custom_dag,
                               dag_opts,
                               matern,
                               daggp_n_threads);
    }
    daggps_alt = daggps;

    init_theta_adapt();
    
    S = arma::chol(Sigma_start, "upper");
    Si = arma::inv(arma::trimatu(S));
    Sigma = S.t() * S;
    Q = Si * Si.t();
    
    // future
    //A = S.t();
    //Aplus = arma::pinv(A);
    //AplusT = Aplus.t();
    
    compute_V();
    
    int nfill = latent_model == 2 ? n : rows_with_missing.n_elem;
    Rw_no_Q = arma::field<arma::mat> (nfill);
    Pblanket_no_Q = arma::field<arma::mat> (nfill);
    
    arma::uvec updater = arma::ones<arma::uvec>(q);
    if(latent_model | Y_needs_filling){
      // first time making markov blanket cache
      cache_blanket_comps(updater);
    }
    
    timings = arma::zeros(10);
  }
};
