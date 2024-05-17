//! This module contains the implementations of the different diffusion processes.
//!
//! The following diffusion processes are implemented:
//! - Cox-Ingersoll-Ross (CIR)
//! dX(t) = theta(mu - X(t))dt + sigma sqrt(X(t))dW(t)
//!
//! - Fractional Cox-Ingersoll-Ross (fCIR)
//! dX(t) = theta(mu - X(t))dt + sigma sqrt(X(t))dW^H(t)
//!
//! - Geometric Brownian Motion (GBM)
//! dX(t) = mu X(t)dt + sigma X(t)dW(t)
//!
//! - Fractional Geometric Brownian Motion (fGBM)
//! dX(t) = mu X(t)dt + sigma X(t)dW^H(t)
//!
//! - Jacobi Process
//! dX(t) = (alpha - beta X(t))dt + sigma (X(t)(1 - X(t)))^(1/2)dW(t)
//!
//! - Fractional Jacobi Process
//! dX(t) = (alpha - beta X(t))dt + sigma (X(t)(1 - X(t)))^(1/2)dW^H(t)
//!
//! - Ornstein-Uhlenbeck (OU)
//! dX(t) = theta(mu - X(t))dt + sigma dW(t)
//!
//! - Fractional Ornstein-Uhlenbeck (fOU)
//! dX(t) = theta(mu - X(t))dt + sigma dW^H(t)

pub mod cir;
pub mod gbm;
pub mod jacobi;
pub mod ou;
