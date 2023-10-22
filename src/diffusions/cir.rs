use crate::{
    noises::{fgn_cholesky, fgn_fft, gn},
    utils::NoiseGenerationMethod,
};
use ndarray::Array1;

pub fn cir(
    theta: f64,
    mu: f64,
    sigma: f64,
    n: usize,
    x0: Option<f64>,
    t: Option<f64>,
    use_sym: Option<bool>,
) -> Vec<f64> {
    let gn = gn::gn(n - 1, Some(t.unwrap_or(1.0)));
    let dt = t.unwrap_or(1.0) / n as f64;

    let mut cir = Array1::<f64>::zeros(n + 1);
    cir[0] = x0.unwrap_or(0.0);

    for (i, dw) in gn.iter().enumerate() {
        let random = match use_sym.unwrap_or(false) {
            true => sigma * (cir[i]).abs().sqrt() * dw,
            false => sigma * (cir[i]).max(0.0).sqrt() * dw,
        };
        cir[i + 1] = theta * (mu - cir[i]) * dt + random
    }

    cir.to_vec()
}

pub fn fcir(
    hurst: f64,
    theta: f64,
    mu: f64,
    sigma: f64,
    n: usize,
    x0: Option<f64>,
    t: Option<f64>,
    method: Option<NoiseGenerationMethod>,
    use_sym: Option<bool>,
) -> Vec<f64> {
    let fgn = match method.unwrap_or(NoiseGenerationMethod::Fft) {
        NoiseGenerationMethod::Fft => fgn_fft::fgn(hurst, n, t.unwrap_or(1.0)),
        NoiseGenerationMethod::Cholesky => fgn_cholesky::fgn(hurst, n - 1, t.unwrap_or(1.0)),
    };
    let dt = t.unwrap_or(1.0) / n as f64;

    let mut fcir = Array1::<f64>::zeros(n + 1);
    fcir[0] = x0.unwrap_or(0.0);

    for (i, dw) in fgn.iter().enumerate() {
        let random = match use_sym.unwrap_or(false) {
            true => sigma * (fcir[i]).abs().sqrt() * dw,
            false => sigma * (fcir[i]).max(0.0) * dw,
        };
        fcir[i + 1] = theta * (mu - fcir[i]) * dt + random
    }

    fcir.to_vec()
}
