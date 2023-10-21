use ndarray::Array1;

use crate::{
    noises::{fgn_cholesky, fgn_fft, gn},
    utils::NoiseGenerationMethod,
};

pub fn jacobi(alpha: f64, beta: f64, sigma: f64, n: usize, t: Option<f64>) -> Vec<f64> {
    let gn = gn::gn(n - 1, t.unwrap_or(1.0));
    let dt = t.unwrap_or(1.0) / n as f64;

    let mut jacobi = Array1::<f64>::zeros(n + 1);
    for (i, dw) in gn.iter().enumerate() {
        jacobi[i + 1] = jacobi[i]
            + (alpha - beta * jacobi[i]) * dt
            + sigma * (jacobi[i] * (1.0 - jacobi[i])).sqrt() * dw
    }

    jacobi.to_vec()
}

pub fn fjacobi(
    hurst: f64,
    theta: f64,
    beta: f64,
    sigma: f64,
    n: usize,
    t: Option<f64>,
    method: Option<NoiseGenerationMethod>,
) -> Vec<f64> {
    let fgn = match method.unwrap_or(NoiseGenerationMethod::Fft) {
        NoiseGenerationMethod::Fft => fgn_fft::fgn(hurst, n, t.unwrap_or(1.0)),
        NoiseGenerationMethod::Cholesky => fgn_cholesky::fgn(hurst, n - 1, t.unwrap_or(1.0)),
    };
    let dt = t.unwrap_or(1.0) / n as f64;

    let mut fjacobi = Array1::<f64>::zeros(n + 1);
    for (i, dw) in fgn.iter().enumerate() {
        fjacobi[i + 1] =
            fjacobi[i] + theta * (beta - fjacobi[i]) * dt + sigma * fjacobi[i].sqrt() * dw
    }

    fjacobi.to_vec()
}
