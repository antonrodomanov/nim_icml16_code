#include <iostream>
#include <random>
#include <functional>
#include <deque>
#include <vector>
#include <algorithm>

#include "optim.h"
#include "Logger.h"
#include "QuadraticFunction.h"

/* ================================================================================================================== */
/* ============================================ IndexSampler ======================================================== */
/* ================================================================================================================== */

class IndexSampler {
public:
    IndexSampler(int N, const std::string& sampling_scheme)
        : N(N), sampling_scheme(sampling_scheme), counter(0)
    {
        if (sampling_scheme == "cyclic") {
        } else if (sampling_scheme == "random") {
            /* prepare uniform random number generator */
            std::random_device rd;
            gen = std::mt19937(rd());
            dis = std::uniform_int_distribution<>(0, N - 1);
        } else if (sampling_scheme == "permute") {
            /* populate `indices` with 0, 1, ..., N-1 for further random shuffling */
            indices.resize(N);
            for (int j = 0; j < N; ++j) indices[j] = j;
        } else {
            fprintf(stderr, "Unknown sampling scheme.\n");
            throw 1;
        }
    }

    int sample()
    {
        /* initialisation */
        int i = -1;

        /* actual sampling */
        if (sampling_scheme == "cyclic") {
            i = counter % N;
        } else if (sampling_scheme == "random") {
            i = dis(gen);
        } else if (sampling_scheme == "permute") {
            if (counter % N == 0) { /* random shuffle every epoch */
                std::random_shuffle(indices.begin(), indices.end());
            }
            i = indices[counter % N];
        }

        /* count this call */
        ++counter;

        /* return result */
        assert(i >= 0 && i < N); // make sure the returned index is valid
        return i;
    }

private:
    int N; // total number of samples
    const std::string& sampling_scheme; // sampling scheme: cyclic, random or permute
    std::mt19937 gen; // random generator (for the "random" sampling scheme)
    std::uniform_int_distribution<> dis; // uniform number generator (for the "random" sampling scheme)
    std::vector<int> indices; // random indices (for the "permute" sampling scheme)
    size_t counter; // total number of samplings made
};

/* ****************************************************************************************************************** */
/* *************************************************** SGD ********************************************************** */
/* ****************************************************************************************************************** */
Eigen::VectorXd SGD(const LogRegOracle& func, Logger& logger, const Eigen::VectorXd& w0, size_t maxiter, double alpha,
                    const std::string& sampling_scheme)
{
    /* assign starting point */
    Eigen::VectorXd w = w0;

    /* initialise index sampler */
    IndexSampler sampler(func.n_samples(), sampling_scheme);

    /* log initial position */
    logger.log(w);

    /* main loop */
    for (size_t iter = 0; iter < maxiter; ++iter) {
        /* select next index */
        int i = sampler.sample();

        /* compute its gradient g_i = nabla f_i(w) */
        Eigen::VectorXd gi = func.single_grad(w, i);

        /* make a step w -= alpha * g_i */
        double epoch = double(iter) / func.n_samples();
        w -= (alpha / (epoch + 1)) * gi;

        /* log current position */
        if (logger.log(w)) break;
    }

    return w;
}

/* ****************************************************************************************************************** */
/* *************************************************** SAG ********************************************************** */
/* ****************************************************************************************************************** */
void sag_update_model(const LogRegOracle& func, int i, const Eigen::VectorXd& w,
                      Eigen::VectorXd& phi_prime, Eigen::VectorXd& g)
{
    /* assign useful variables */
    const int N = func.n_samples();
    const Eigen::MatrixXd& Z = func.Z;

    /* take i-th training sample */
    Eigen::VectorXd zi = Z.row(i).transpose();

    /* compute phi_prime_new = phi'(z_i' * w) */
    double phi_prime_new = func.phi_prime(zi.dot(w));

    /* update g: g += 1/N delta_phi_prime * z_i */
    double delta_phi_prime = phi_prime_new - phi_prime(i);
    g += (1.0 / N) * delta_phi_prime * zi;

    /* update model */
    phi_prime(i) = phi_prime_new;
}

Eigen::VectorXd SAG(const LogRegOracle& func, Logger& logger, const Eigen::VectorXd& w0, size_t maxiter, double alpha,
                    const std::string& sampling_scheme, const std::string& init_scheme)
{
    /* assign useful variables */
    const int N = func.n_samples();
    const int D = w0.size();
    const double lambda = func.lambda;

    /* assign starting point */
    Eigen::VectorXd w = w0;

    /* initialise index sampler */
    IndexSampler sampler(N, sampling_scheme);

    /* initialisation */
    Eigen::VectorXd phi_prime = Eigen::VectorXd::Zero(N); // coefficients phi_prime(i) = phi'(z_i' * v_i)

    Eigen::VectorXd g = Eigen::VectorXd::Zero(D); // average gradient g = 1/N sum_i nabla f_i(v_i)

    if (init_scheme == "self-init") {
        /* nothing to do here, everything will be done in the main loop */
    } else if (init_scheme == "full") {
        /* initialise all the components of the model at w0 */
        for (int i = 0; i < N; ++i) {
            /* update the current component of the model */
            sag_update_model(func, i, w0, phi_prime, g);

            /* don't cheat, call the logger because initialisation counts too */
            if (logger.log(w0)) break;
        }
    } else {
        fprintf(stderr, "Unknown initialisation scheme.\n");
        throw 1;
    }

    /* log initial position */
    logger.log(w);

    /* main loop */
    for (size_t iter = 0; iter < maxiter; ++iter) {
        /* select next index */
        int i = sampler.sample();

        /* update the i-th component of the model */
        sag_update_model(func, i, w, phi_prime, g);

        /* make a step w -= alpha * (g + lambda * w) */
        w = func.prox1(w - alpha * (g + lambda * w), alpha);

        /* log current position */
        if (logger.log(w)) break;
    }

    return w;
}

/* ****************************************************************************************************************** */
/* *************************************************** SO2 ********************************************************** */
/* ****************************************************************************************************************** */
void so2_update_model(const LogRegOracle& func, int i, const Eigen::VectorXd& w,
                      Eigen::VectorXd& mu, Eigen::VectorXd& phi_prime, Eigen::VectorXd& phi_double_prime,
                      Eigen::VectorXd& g, Eigen::VectorXd& p,
                      Eigen::MatrixXd& H)
{
    /* assign useful variables */
    const int N = func.n_samples();
    const Eigen::MatrixXd& Z = func.Z;

    /* take i-th training sample */
    Eigen::VectorXd zi = Z.row(i).transpose();

    /* compute new mu_i = z_i' * v_i where v_i = w */
    double mu_new = zi.dot(w);

    /* compute phi' and phi'' at mu_i */
    double phi_prime_new = func.phi_prime(mu_new);
    double phi_double_prime_new = func.phi_double_prime(mu_new);

    /* update g: g += 1/N delta_phi_prime z_i */
    double delta_phi_prime = phi_prime_new - phi_prime(i);
    g += (1.0 / N) * delta_phi_prime * zi;

    /* update p: p += 1/N (phi_double_prime_new * mu_new - phi_double_prime * mu) * z_i */
    double delta_phi_double_prime_mu = phi_double_prime_new * mu_new - phi_double_prime(i) * mu(i);
    p += (1.0 / N) * delta_phi_double_prime_mu * zi;

    /* update H */
    double delta_phi_double_prime = phi_double_prime_new - phi_double_prime(i);
    H.selfadjointView<Eigen::Upper>().rankUpdate(zi, delta_phi_double_prime/N);

    /* update B using Sherman-Morrison-Woodbury formula (rank-1 update) */
    //Eigen::VectorXd bzi = B.selfadjointView<Eigen::Upper>() * zi;
    //double coef = delta_phi_double_prime / (N + delta_phi_double_prime * zi.dot(bzi));
    //B.selfadjointView<Eigen::Upper>().rankUpdate(bzi, -coef);

    /* update bgmp: bgmp += [1/N (delta_phi_prime - delta_phi_double_prime_mu) - coef * bzi' (g_new - p_new)] * bzi */
    //bgmp += ((1.0 / N) * (delta_phi_prime - delta_phi_double_prime_mu) - coef * bzi.dot(g - p)) * bzi;
    //bgmp = B.selfadjointView<Eigen::Upper>() * (g - p);
    //Eigen::LLT<Eigen::MatrixXd, Eigen::Upper> llt;
    //llt.compute(H);
    //bgmp = llt.solve(g - p);

    /* update model */
    mu(i) = mu_new;
    phi_prime(i) = phi_prime_new;
    phi_double_prime(i) = phi_double_prime_new;
}

Eigen::VectorXd SO2(const LogRegOracle& func, Logger& logger, const Eigen::VectorXd& w0, size_t maxiter, double alpha,
                    const std::string& sampling_scheme, const std::string& init_scheme, bool exact)
{
    /* assign useful variables */
    const int N = func.n_samples();
    const int D = w0.size();
    const double lambda = func.lambda;

    /* assign starting point */
    Eigen::VectorXd w = w0;

    /* initialisation */
    Eigen::VectorXd mu = Eigen::VectorXd::Zero(N); // coefficients mu_i = z_i' * v_i
    Eigen::VectorXd phi_prime = Eigen::VectorXd::Zero(N); // coefficients phi_prime(i) = phi'(mu_i)
    Eigen::VectorXd phi_double_prime = Eigen::VectorXd::Zero(N); // coefficients phi_doube_prime(i) = phi''(mu_i)

    Eigen::VectorXd g = Eigen::VectorXd::Zero(D); // average gradient g = 1/N sum_i nabla f_i(v_i)
    Eigen::VectorXd p = Eigen::VectorXd::Zero(D); // vector p = 1/N sum_i nabla^2 f_i(v_i) v_i
    //Eigen::VectorXd bgmp = Eigen::VectorXd::Zero(D); // vector bgmp = B * (g - p)

    Eigen::MatrixXd H = lambda * Eigen::MatrixXd::Identity(D, D); // average hessian H = (1/N sum_i nabla^2 f_i(v_i))
    //Eigen::MatrixXd B = (1.0 / lambda) * Eigen::MatrixXd::Identity(D, D); // inverse average hessian B = (1/N sum_i nabla^2 f_i(v_i))^{-1}

    if (init_scheme == "self-init") {
        /* nothing to do here, everything will be done in the main loop */
    } else if (init_scheme == "full") {
        /* initialise all the components of the model at w0 */
        for (int i = 0; i < N; ++i) {
            /* update the current component of the model */
            so2_update_model(func, i, w0, mu, phi_prime, phi_double_prime, g, p, H);

            /* don't cheat, call the logger because initialisation counts too */
            if (logger.log(w0)) break;
        }
    } else {
        fprintf(stderr, "Unknown initialisation scheme.\n");
        throw 1;
    }

    /* initialise index sampler */
    IndexSampler sampler(N, sampling_scheme);

    /* log initial position */
    logger.log(w);

    /* main loop */
    for (size_t iter = 0; iter < maxiter; ++iter) {
        /* choose index */
        int i = sampler.sample();

        /* update the i-th component of the model */
        so2_update_model(func, i, w, mu, phi_prime, phi_double_prime, g, p, H);

        /* make a step w -= alpha * (w + B * (g - p)) */
        Eigen::VectorXd b = g - p;
        QuadraticFunction mk = QuadraticFunction(H, b, func.lambda1);
        double norm_g = (w - func.prox1(w - g, 1)).lpNorm<2>();
        double tol_fgm;
        if (!exact) {
            tol_fgm = std::min(1.0, pow(norm_g, 0.5)) * norm_g;
        } else {
            tol_fgm = 1e-10;
        }
        if (int(iter) < func.n_samples()) {
            tol_fgm = 1;
        }
        size_t maxiter_fgm = 10000;
        Eigen::VectorXd w_bar = fgm(mk, w, maxiter_fgm, tol_fgm);
        w += alpha * (w_bar - w);

        /* log current position */
        if (logger.log(w)) break;
    }

    return w;
}

/* ****************************************************************************************************************** */
/* ************************************************** Newton ******************************************************** */
/* ****************************************************************************************************************** */

Eigen::VectorXd newton(const LogRegOracle& func, Logger& logger, const Eigen::VectorXd& w0, size_t maxiter, bool exact)
{
    /* some parameters */
    const size_t maxiter_fgm = 10000;

    /* assign starting point */
    Eigen::VectorXd w = w0;

    /* log initial position */
    logger.log(w);

    /* initialisation */
    size_t n_full_calls = 0;
    Eigen::VectorXd g = func.full_grad(w); // gradient
    Eigen::MatrixXd H = func.full_hess(w); // Hessian
    ++n_full_calls;

    /* main loop */
    for (size_t iter = 0; iter < maxiter; ++iter) {
        Eigen::VectorXd b = g - H.selfadjointView<Eigen::Upper>() * w;
        QuadraticFunction mk = QuadraticFunction(H, b, func.lambda1);
        double tol_fgm;
        if (!exact) {
            double norm_g = (w - func.prox1(w - g, 1)).lpNorm<2>();
            tol_fgm = std::min(1.0, sqrt(norm_g)) * norm_g;
        } else {
            tol_fgm = 1e-10;
        }
        w = fgm(mk, w, maxiter_fgm, tol_fgm);

        /* call function at new point */
        g = func.full_grad(w);
        H = func.full_hess(w);
        ++n_full_calls;

        /* log current position */
        if (logger.log(w, n_full_calls)) break;
    }

    return w;
}

/* ****************************************************************************************************************** */
/* *************************************************** FGM ********************************************************** */
/* ****************************************************************************************************************** */

Eigen::VectorXd fgm(const CompositeFunction& func, const Eigen::VectorXd& x0, size_t maxiter, double tol, double L0)
{
    /* assign starting point */
    Eigen::VectorXd x = x0;

    /* initialisation */
    double A = 0;
    double L = L0;
    Eigen::VectorXd v_bar = x0;

    /* main loop */
    Eigen::VectorXd T, gfT, gy, y, gfy;
    size_t iter;
    for (iter = 0; iter < maxiter; ++iter) {
        Eigen::VectorXd v = func.prox1(v_bar, A);
        double a;
        while (true) {
            double b = 1/L;
            a = b + sqrt(b*b + 2*b*A);
            y = (A * x + a * v) / (A + a);
            gfy = func.full_grad(y);
            T = func.prox1(y - (1/L)*gfy, 1/L);
            gy = L*(y - T); // composite gradient

            if (gy.lpNorm<Eigen::Infinity>() < tol) {
                break;
            }

            gfT = func.full_grad(T);

            Eigen::VectorXd phi_prime = gy - (gfy - gfT);
            if (phi_prime.dot(y - T) >= (1/L)*phi_prime.squaredNorm()) {
                break;
            }

            L = 2 * L;
        }

        if (gy.lpNorm<Eigen::Infinity>() < tol) {
            x = T;
            break;
        }

        x = T;
        v_bar = v_bar - a * gfT;
        A = A + a;

        L = std::max(L0, L / 2);
    }

    //fprintf(stderr, "FGM finished in %zu iterations, norm_g=%g\n", iter, gy.lpNorm<Eigen::Infinity>());

    return x;
}

/* ****************************************************************************************************************** */
/* ******************************************** Conjugate gradient ************************************************** */
/* ****************************************************************************************************************** */

Eigen::VectorXd cg(const std::function<Eigen::VectorXd(const Eigen::VectorXd&)>& matvec,
                   const Eigen::VectorXd& b, const Eigen::VectorXd& x0, double tol)
{
    /* assign starting point */
    Eigen::VectorXd x = x0;

    /* initialisation */
    size_t maxiter = b.size(); // maximum number of iteration (equals n by default)

    Eigen::VectorXd r = matvec(x) - b; // residual
    Eigen::VectorXd d = -r; // direction
    double norm_r = r.lpNorm<Eigen::Infinity>(); // residual infinity-norm
    double r2 = r.dot(r); // residual 2-norm squared

    /* main loop */
    size_t iter = 0;
    while (iter < maxiter && norm_r > tol) {
        /* compute matrix-vector product */
        Eigen::VectorXd ad = matvec(d);

        /* update current point and residual */
        double alpha = r2 / (d.dot(ad));
        x += alpha * d;
        r += alpha * ad;

        /* update direction */
        double r2_new = r.dot(r);
        double beta = r2_new / r2;
        d = -r + beta * d;

        /* prepare for next iteration */
        ++iter;
        r2 = r2_new;
        norm_r = r.lpNorm<Eigen::Infinity>();
    }

    return x;
}

/* ****************************************************************************************************************** */
/* ******************************************* Hessian-free Newton ************************************************** */
/* ****************************************************************************************************************** */

Eigen::VectorXd HFN(const LogRegOracle& func, Logger& logger, const Eigen::VectorXd& w0, size_t maxiter, double c1)
{
    /* assign starting point */
    Eigen::VectorXd w = w0;

    /* log initial position */
    logger.log(w);

    /* initialisation */
    size_t n_full_calls = 0; // total number of function calls

    double f; // function value
    Eigen::VectorXd g; // gradient
    f = func.full_val_grad(w, g);
    ++n_full_calls;

    Eigen::VectorXd d = Eigen::VectorXd::Zero(w.size()); // direction

    LogRegHessVec hv = func.hessvec(); // for Hessian-vector products

    /* main loop */
    for (size_t iter = 0; iter < maxiter; ++iter) {
        /* calculate direction d = -H^{-1} g approximately using CG */
        double norm_g = g.lpNorm<Eigen::Infinity>();
        double cg_tol = std::min(0.5, sqrt(norm_g)) * norm_g;
        hv.prepare(w); // prepare for computing multiple Hessian-vector products at current point
        auto matvec = [&hv](const Eigen::VectorXd& d) { return hv.calculate(d); };
        double gtd;
        while (true) {
            d = cg(matvec, -g, d, cg_tol);

            /* ensure the returned `d` is a *descent* direction */
            gtd = g.dot(d); // directional derivative
            if (gtd <= 0.0) break;

            /* otherwise, increase tolerance for CG and recompute `d` */
            cg_tol /= 10.0;
            fprintf(stderr, "not a descent direction, increase CG tolerance: cg_tol=%g\n", cg_tol);
        }

        /* backtrack if needed */
        double alpha = 1.0; // initial step length
        assert(gtd <= 0.0); // descent direction
        while (true) {
            /* make a step w += alpha * d */
            Eigen::VectorXd w_new = w + alpha * d;

            /* call function at new point */
            double f_new = func.full_val_grad(w_new, g);
            ++n_full_calls;

            /* check Armijo condition */
            if (f_new <= f + c1 * alpha * gtd || norm_g < 1e-6) { // always use unit step length when close to optimum
                w = w_new;
                f = f_new;
                break;
            }

            /* if not satisfied, halve step length */
            alpha /= 2;
            fprintf(stderr, "backtrack (alpha=%g)...\n", alpha);
        }

        /* log current position */
        if (logger.log(w, n_full_calls)) break;
    }

    return w;
}

/* ****************************************************************************************************************** */
/* *************************************************** BFGS ********************************************************* */
/* ****************************************************************************************************************** */

Eigen::VectorXd BFGS(const LogRegOracle& func, Logger& logger, const Eigen::VectorXd& w0, size_t maxiter, double c1)
{
    /* assign starting point */
    Eigen::VectorXd w = w0;

    /* log initial position */
    logger.log(w);

    /* initialisation */
    size_t n_full_calls = 0;

    double f; // function value
    Eigen::VectorXd g; // gradient
    f = func.full_val_grad(w, g);
    ++n_full_calls;

    Eigen::MatrixXd B = Eigen::MatrixXd::Identity(w.size(), w.size()); // BFGS approximation for the Hessian

    /* auxiliary variables */
    double f_new;
    Eigen::VectorXd w_new, g_new;

    /* main loop */
    for (size_t iter = 0; iter < maxiter; ++iter) {
        /* calculate direction d = -B*g */
        Eigen::VectorXd d = B.selfadjointView<Eigen::Upper>() * (-g);

        /* backtrack if needed */
        double gtd = g.dot(d); // directional derivative
        assert(gtd <= 0.0);
        double norm_g = g.lpNorm<Eigen::Infinity>();
        double alpha = 1.0; // initial step length
        while (true) {
            /* make a step w += alpha * d */
            w_new = w + alpha * d;

            /* call function at new point */
            f_new = func.full_val_grad(w_new, g_new);
            ++n_full_calls;

            /* check Armijo condition */
            if (f_new <= f + c1 * alpha * gtd || norm_g < 1e-6) { // always use unit step length when close to optimum
                break;
            }

            /* if not satisfied, halve step length */
            alpha /= 2;
            fprintf(stderr, "backtrack (alpha=%g)...\n", alpha);
        }

        /* update B: B_new = (I - rho*y*s')'*B*(I - rho*y*s') + rho*s*s', where rho=1/(y'*s) */
        Eigen::VectorXd y = g_new - g;
        Eigen::VectorXd s = w_new - w;
        assert(y.dot(s) > 0); // this should hold for strongly convex functions
        double rho = 1.0 / y.dot(s);
        Eigen::VectorXd by = B.selfadjointView<Eigen::Upper>() * y;
        B.selfadjointView<Eigen::Upper>().rankUpdate(by, s, -rho); // symmetric rank-2 update
        double coef = rho * (rho * y.dot(by) + 1);
        B.selfadjointView<Eigen::Upper>().rankUpdate(s, coef); // symmetric rank-1 update

        /* prepare for next iteration */
        w = w_new;
        f = f_new;
        g = g_new;

        /* log current position */
        if (logger.log(w, n_full_calls)) break;
    }

    return w;
}

/* ****************************************************************************************************************** */
/* ************************************************** L-BFGS ******************************************************** */
/* ****************************************************************************************************************** */

/* L-BFGS two-loop recursion */
Eigen::VectorXd lbfgs_prod(std::deque<std::pair<Eigen::VectorXd, Eigen::VectorXd>> ys_hist, const Eigen::VectorXd& q, double gamma0)
{
    /* auxiliary variables */
    size_t m = ys_hist.size();
    Eigen::VectorXd y, s;
    Eigen::VectorXd coefs1 = Eigen::VectorXd::Zero(m);
    double rho, coef2;
    int i;

    /* assign initial vector */
    Eigen::VectorXd z = q;

    /* go back */
    i = m - 1;
    for (auto it = ys_hist.rbegin(); it != ys_hist.rend(); ++it) {
        std::tie(y, s) = *it;
        rho = 1.0 / y.dot(s);
        coefs1(i) = rho * s.dot(z);
        z -= coefs1(i) * y;
        --i;
    }

    /* multiply by initial matrix `gamma0*I` */
    z *= gamma0;

    /* go forward */
    i = 0;
    for (auto it = ys_hist.begin(); it != ys_hist.end(); ++it) {
        std::tie(y, s) = *it;
        rho = 1.0 / y.dot(s);
        coef2 = rho * y.dot(z);
        z += (coefs1(i) - coef2) * s;
        ++i;
    }

    return z;
}

Eigen::VectorXd LBFGS(const LogRegOracle& func, Logger& logger, const Eigen::VectorXd& w0, size_t maxiter, size_t m, double c1)
{
    /* assign starting point */
    Eigen::VectorXd w = w0;

    /* log initial position */
    logger.log(w);

    /* initialisation */
    size_t n_full_calls = 0;

    double f; // function value
    Eigen::VectorXd g; // gradient
    f = func.full_val_grad(w, g);
    ++n_full_calls;

    std::deque<std::pair<Eigen::VectorXd, Eigen::VectorXd>> ys_hist; // L-BFGS history
    Eigen::VectorXd d; // direction

    /* auxiliary variables */
    double f_new;
    Eigen::VectorXd w_new, g_new;

    /* main loop */
    for (size_t iter = 0; iter < maxiter; ++iter) {
        /* calculate direction */
        double norm_g = g.lpNorm<Eigen::Infinity>();
        Eigen::VectorXd d;
        if (iter == 0) { // first iteration
            d = -g / norm_g;
        } else { // can use L-BFGS history
            /* calculate the coefficient for the initial matrix */
            Eigen::VectorXd y_old, s_old;
            std::tie(y_old, s_old) = ys_hist.back();
            double gamma0 = (y_old.dot(s_old)) / (y_old.dot(y_old)); // Barzilai-Borwein initialisation

            /* use L-BFGS two-loop recursion */
            d = lbfgs_prod(ys_hist, -g, gamma0);
        }

        /* backtrack if needed */
        double gtd = g.dot(d); // directional derivative
        assert(gtd <= 0.0);
        double alpha = 1.0; // initial step length
        while (true) {
            /* make a step w += alpha * d */
            w_new = w + alpha * d;

            /* call function at new point */
            f_new = func.full_val_grad(w_new, g_new);
            ++n_full_calls;

            /* check Armijo condition */
            if (f_new <= f + c1 * alpha * gtd || norm_g < 1e-6) { // always use unit step length when close to optimum
                break;
            }

            /* if not satisfied, halve step length */
            alpha /= 2;
            fprintf(stderr, "backtrack (alpha=%g)...\n", alpha);
        }

        /* update L-BFGS history */
        Eigen::VectorXd y = g_new - g;
        Eigen::VectorXd s = w_new - w;
        assert(y.dot(s) > 0); // this should hold for strongly convex functions
        if (ys_hist.size() >= m) ys_hist.pop_front();
        ys_hist.push_back(std::make_pair(y, s));

        /* prepare for next iteration */
        w = w_new;
        f = f_new;
        g = g_new;

        /* log current position */
        if (logger.log(w, n_full_calls)) break;
    }

    return w;
}
