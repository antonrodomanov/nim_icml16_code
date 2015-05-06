#include <iostream>
#include <random>
#include <functional>

#include "optim.h"
#include "Logger.h"

/* ****************************************************************************************************************** */
/* *************************************************** SGD ********************************************************** */
/* ****************************************************************************************************************** */
Eigen::VectorXd SGD(const LogRegOracle& func, Logger& logger, const Eigen::VectorXd& w0, double alpha, size_t maxiter)
{
    /* prepare random number generator */
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, func.n_samples() - 1);

    /* assign starting point */
    Eigen::VectorXd w = w0;

    /* log initial position */
    logger.log(w);

    /* main loop */
    for (size_t iter = 0; iter < maxiter; ++iter) {
        /* select random sample i */
        int i = dis(gen);

        /* compute its gradient g_i = nabla f_i(w) */
        Eigen::VectorXd gi = func.single_grad(w, i);

        /* make a step w -= alpha * g_i */
        w -= alpha * gi;

        /* log current position */
        if (logger.log(w)) break;
    }

    return w;
}

/* ****************************************************************************************************************** */
/* *************************************************** SAG ********************************************************** */
/* ****************************************************************************************************************** */
Eigen::VectorXd SAG(const LogRegOracle& func, Logger& logger, const Eigen::VectorXd& w0, double alpha, size_t maxiter)
{
    /* assign useful variables */
    const int N = func.n_samples();
    const int D = w0.size();
    const double lambda = func.lambda;
    const Eigen::MatrixXd& Z = func.Z;

    /* prepare random number generator */
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, N - 1);

    /* assign starting point */
    Eigen::VectorXd w = w0;

    /* initialisation */
    Eigen::VectorXd phi_prime = Eigen::VectorXd::Zero(N); // coefficients phi_prime(i) = phi'(z_i' * v_i)

    Eigen::VectorXd g = Eigen::VectorXd::Zero(D); // average gradient g = 1/N sum_i nabla f_i(v_i)

    /* log initial position */
    logger.log(w);

    /* main loop */
    for (size_t iter = 0; iter < maxiter; ++iter) {
        /* select random sample i */
        int i = dis(gen);

        /* take i-th training sample */
        Eigen::VectorXd zi = Z.row(i).transpose();

        /* compute phi_prime_new = phi'(z_i' * w) */
        double phi_prime_new = func.phi_prime(zi.dot(w));

        /* update g: g += 1/N delta_phi_prime * z_i */
        double delta_phi_prime = phi_prime_new - phi_prime(i);
        g += (1.0 / N) * delta_phi_prime * zi;

        /* update model */
        phi_prime(i) = phi_prime_new;

        /* make a step w -= alpha * (g + lambda * w) */
        w -= alpha * (g + lambda * w);

        /* log current position */
        if (logger.log(w)) break;
    }

    return w;
}

/* ****************************************************************************************************************** */
/* *************************************************** SO2 ********************************************************** */
/* ****************************************************************************************************************** */
Eigen::VectorXd SO2(const LogRegOracle& func, Logger& logger, const Eigen::VectorXd& w0, size_t maxiter)
{
    /* assign useful variables */
    const int N = func.n_samples();
    const int D = w0.size();
    const double lambda = func.lambda;
    const Eigen::MatrixXd& Z = func.Z;

    /* assign starting point */
    Eigen::VectorXd w = w0;

    /* initialisation */
    Eigen::VectorXd mu = Eigen::VectorXd::Zero(N); // coefficients mu_i = z_i' * v_i
    Eigen::VectorXd phi_prime = Eigen::VectorXd::Zero(N); // coefficients phi_prime(i) = phi'(mu_i)
    Eigen::VectorXd phi_double_prime = Eigen::VectorXd::Zero(N); // coefficients phi_doube_prime(i) = phi''(mu_i)

    Eigen::VectorXd g = Eigen::VectorXd::Zero(D); // average gradient g = 1/N sum_i nabla f_i(v_i)
    Eigen::VectorXd p = Eigen::VectorXd::Zero(D); // vector p = 1/N sum_i nabla^2 f_i(v_i) v_i
    Eigen::VectorXd bgmp = Eigen::VectorXd::Zero(D); // vector bgmp = B * (g - p)

    Eigen::MatrixXd B = (1.0 / lambda) * Eigen::MatrixXd::Identity(D, D); // inverse average hessian B = (1/N sum_i nabla^2 f_i(v_i))^{-1}

    int i = -1; // sample index; start with -1 because the first one will be (i+1) % N = 0
    double alpha = 1.0; // step length (always use unit step length for now)

    /* log initial position */
    logger.log(w);

    /* main loop */
    for (size_t iter = 0; iter < maxiter; ++iter) {
        /* choose index; use cyclic order */
        i = (i + 1) % N;

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

        /* update B using Sherman-Morrison-Woodbury formula (rank-1 update) */
        double delta_phi_double_prime = phi_double_prime_new - phi_double_prime(i);
        Eigen::VectorXd bzi = B.selfadjointView<Eigen::Upper>() * zi;
        double coef = delta_phi_double_prime / (N + delta_phi_double_prime * zi.dot(bzi));
        B.selfadjointView<Eigen::Upper>().rankUpdate(bzi, -coef);

        /* update bgmp: bgmp += [1/N (delta_phi_prime - delta_phi_double_prime_mu) - coef * bzi' (g_new - p_new)] * bzi */
        bgmp += ((1.0 / N) * (delta_phi_prime - delta_phi_double_prime_mu) - coef * bzi.dot(g - p)) * bzi;

        /* update model */
        mu(i) = mu_new;
        phi_prime(i) = phi_prime_new;
        phi_double_prime(i) = phi_double_prime_new;

        /* make a step w -= alpha * (w + B * (g - p)) */
        w -= alpha * (w + bgmp);

        /* log current position */
        if (logger.log(w)) break;
    }

    return w;
}

/* ****************************************************************************************************************** */
/* ************************************************** Newton ******************************************************** */
/* ****************************************************************************************************************** */

Eigen::VectorXd newton(const LogRegOracle& func, Logger& logger, const Eigen::VectorXd& w0, size_t maxiter, double c1)
{
    /* assign starting point */
    Eigen::VectorXd w = w0;

    /* log initial position */
    logger.log(w);

    /* initialisation */
    size_t n_full_calls = 0;

    double f = func.full_val(w); // function value
    Eigen::VectorXd g = func.full_grad(w); // gradient
    Eigen::MatrixXd H = func.full_hess(w); // Hessian
    ++n_full_calls;

    Eigen::LLT<Eigen::MatrixXd, Eigen::Upper> llt; // for calculating Cholesky decomposition of H

    /* main loop */
    for (size_t iter = 0; iter < maxiter; ++iter) {
        /* compute Cholesky decomposition H=L*L.T */
        llt.compute(H);

        /* calculate direction d = -H^{-1} g */
        Eigen::VectorXd d = llt.solve(-g);

        /* backtrack if needed */
        double gtd = g.dot(d); // directional derivative
        assert(gtd <= 0.0);
        double norm_g = g.lpNorm<Eigen::Infinity>();
        double alpha = 1.0; // initial step length
        while (true) {
            /* make a step w += alpha * d */
            Eigen::VectorXd w_new = w + alpha * d;

            /* call function at new point */
            double f_new = func.full_val(w_new);
            g = func.full_grad(w_new);
            H = func.full_hess(w_new);
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
        //fprintf(stderr, "iter=%zu, alpha=%g, f=%g, norm_g=%g\n", iter, alpha, f, g.lpNorm<Eigen::Infinity>());
        if (logger.log(w, n_full_calls)) break;
    }

    return w;
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

    double f = func.full_val(w); // function value
    Eigen::VectorXd g = func.full_grad(w); // gradient
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
            double f_new = func.full_val(w_new);
            g = func.full_grad(w_new);
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

    double f = func.full_val(w); // function value
    Eigen::VectorXd g = func.full_grad(w); // gradient
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
            f_new = func.full_val(w_new);
            g_new = func.full_grad(w_new);
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
