use crate::{
    noises::{fgn_cholesky, fgn_fft},
    utils::NoiseGenerationMethod,
};
use nalgebra::RowDVector;

pub fn fbm(
    n: usize,
    hurst: f64,
    t: Option<f64>,
    method: Option<NoiseGenerationMethod>,
) -> Vec<f64> {
    let fgn = match method.unwrap_or(NoiseGenerationMethod::Fft) {
        NoiseGenerationMethod::Fft => fgn_fft::fgn(hurst, n, t.unwrap_or(1.0)),
        NoiseGenerationMethod::Cholesky => fgn_cholesky::fgn(hurst, n - 1, t.unwrap_or(1.0)),
    };

    let mut fbm = RowDVector::<f64>::zeros(n);
    fbm[0] = 0.0;

    for i in 1..n {
        fbm[i] = fbm[i - 1] + fgn[i - 1];
    }

    fbm.data.as_vec().clone()
}
