use crate::{
    noises::{fgn_cholesky, fgn_fft, gn},
    utils::NoiseGenerationMethod,
};
use ndarray::Array1;

pub fn ou(mu: f64, sigma: f64, theta: f64, n: usize, x0: Option<f64>, t: Option<f64>) -> Vec<f64> {
    let gn = gn::gn(n - 1, Some(t.unwrap_or(1.0)));
    let dt = t.unwrap_or(1.0) / n as f64;

    let mut ou = Array1::<f64>::zeros(n + 1);
    ou[0] = x0.unwrap_or(0.0);

    for (i, dw) in gn.iter().enumerate() {
        ou[i + 1] = ou[i] + theta * (mu - ou[i]) * dt + sigma * dw
    }

    ou.to_vec()
}

pub fn fou(
    hurst: f64,
    mu: f64,
    sigma: f64,
    theta: f64,
    n: usize,
    x0: Option<f64>,
    t: Option<f64>,
    method: Option<NoiseGenerationMethod>,
) -> Vec<f64> {
    let fgn = match method.unwrap_or(NoiseGenerationMethod::Fft) {
        NoiseGenerationMethod::Fft => fgn_fft::fgn(hurst, n, t.unwrap_or(1.0)),
        NoiseGenerationMethod::Cholesky => fgn_cholesky::fgn(hurst, n - 1, t.unwrap_or(1.0)),
    };
    let dt = t.unwrap_or(1.0) / n as f64;

    let mut fou = Array1::<f64>::zeros(n + 1);
    fou[0] = x0.unwrap_or(0.0);

    for (i, dw) in fgn.iter().enumerate() {
        fou[i + 1] = fou[i] + theta * (mu - fou[i]) * dt + sigma * dw
    }

    fou.to_vec()
}
