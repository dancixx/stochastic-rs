use crate::{noises::gn::gn, utils::NoiseGenerationMethod};
use ndarray::Array1;

pub fn gbm(mu: f64, sigma: f64, n: usize, t: Option<f64>, x0: Option<f64>) -> Vec<f64> {
    let gn = gn(n - 1, t.unwrap_or(1.0));
    let dt = t.unwrap_or(1.0) / n as f64;

    let mut gbm = Array1::<f64>::zeros(n + 1);
    gbm[0] = x0.unwrap_or(100.0);

    for (i, dw) in gn.iter().enumerate() {
        gbm[i + 1] = gbm[i] + mu * gbm[i] * dt + sigma * gbm[i] * dw
    }

    gbm.to_vec()
}

pub fn fgbm(
    hurst: f64,
    mu: f64,
    sigma: f64,
    n: usize,
    t: Option<f64>,
    x0: Option<f64>,
    method: Option<NoiseGenerationMethod>,
) -> Vec<f64> {
    let gn = match method.unwrap_or(NoiseGenerationMethod::Fft) {
        NoiseGenerationMethod::Fft => crate::noises::fgn_fft::fgn(hurst, n - 1, t.unwrap_or(1.0)),
        NoiseGenerationMethod::Cholesky => {
            crate::noises::fgn_cholesky::fgn(hurst, n - 1, t.unwrap_or(1.0))
        }
    };
    let dt = t.unwrap_or(1.0) / n as f64;

    let mut fgbm = Array1::<f64>::zeros(n + 1);
    fgbm[0] = x0.unwrap_or(100.0);

    for (i, dw) in gn.iter().enumerate() {
        fgbm[i + 1] = fgbm[i] + mu * fgbm[i] * dt + sigma * fgbm[i] * dw
    }

    fgbm.to_vec()
}
