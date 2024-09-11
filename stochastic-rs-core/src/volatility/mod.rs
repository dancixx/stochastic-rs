//! This module contains the implementations of various diffusion processes.
//!
//! The following diffusion processes are implemented:
//!
//! - Duffie-Kan Model
//!   - The Duffie-Kan model is a multifactor interest rate model incorporating correlated Brownian motions.
//!   - SDE: `dr(t) = (a1 * r(t) + b1 * x(t) + c1) * dt + sigma1 * (alpha * r(t) + beta * x(t) + gamma) * dW_r(t)`
//!   - SDE: `dx(t) = (a2 * r(t) + b2 * x(t) + c2) * dt + sigma2 * (alpha * r(t) + beta * x(t) + gamma) * dW_x(t)`
//!   - where Corr(W_r(t), W_x(t)) = rho
//!
//! - **Heston Model**
//!   - A stochastic volatility model used to describe the evolution of the volatility of an underlying asset.
//!   - SDE: `dS(t) = mu * S(t) * dt + S(t) * sqrt(V(t)) * dW_1(t)`
//!   - SDE: `dV(t) = kappa * (theta - V(t)) * dt + eta * sqrt(V(t)) * dW_2(t)`
//!
//! - SABR (Stochastic Alpha, Beta, Rho) Model
//! - Widely used in financial mathematics for modeling stochastic volatility.
//!   - SDE: `dF(t) = V(t) * F(t)^beta * dW_F(t)`
//!   - SDE: `dV(t) = alpha * V(t) * dW_V(t)`
//!   - where Corr(W_F(t), W_V(t)) = rho
//!
//! - **Vasicek Model**
//!   - An Ornstein-Uhlenbeck process used to model interest rates.
//!   - SDE: `dX(t) = theta * (mu - X(t)) * dt + sigma * dW(t)`
//!
//! - **Fractional Vasicek (fVasicek) Model**
//!   - Incorporates fractional Brownian motion into the Vasicek model.
//!   - SDE: `dX(t) = theta * (mu - X(t)) * dt + sigma * dW^H(t)`
//!
//! Each process has its own module and functions to generate sample paths.

pub mod heston;
pub mod sabr;
