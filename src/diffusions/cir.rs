use ndarray::Array1;

use crate::{
    noises::{fgn_cholesky, fgn_fft, gn},
    utils::NoiseGenerationMethod,
};

pub fn cir(theta: f64, beta: f64, sigma: f64, n: usize, t: Option<f64>) -> Vec<f64> {
    let gn = gn::gn(n - 1, Some(t.unwrap_or(1.0)));
    let dt = t.unwrap_or(1.0) / n as f64;

    let mut cir = Array1::<f64>::zeros(n + 1);
    for (i, dw) in gn.iter().enumerate() {
        cir[i + 1] = theta * (beta - cir[i]) * dt + sigma * cir[i].sqrt() * dw
    }

    cir.to_vec()
}

pub fn fcir(
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

    let mut fcir = Array1::<f64>::zeros(n + 1);
    for (i, dw) in fgn.iter().enumerate() {
        fcir[i + 1] = theta * (beta - fcir[i]) * dt + sigma * fcir[i].sqrt() * dw
    }

    fcir.to_vec()
}
