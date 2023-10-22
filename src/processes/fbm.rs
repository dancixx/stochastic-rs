use crate::{
    noises::{fgn_cholesky, fgn_fft},
    utils::NoiseGenerationMethod,
};
use ndarray::Array1;

pub fn fbm(
    hurst: f64,
    n: usize,
    t: Option<f64>,
    method: Option<NoiseGenerationMethod>,
) -> Vec<f64> {
    if !(0.0..1.0).contains(&hurst) {
        panic!("Hurst parameter must be in (0, 1)")
    }

    let fgn = match method.unwrap_or(NoiseGenerationMethod::Fft) {
        NoiseGenerationMethod::Fft => fgn_fft::fgn(hurst, n, t.unwrap_or(1.0)),
        NoiseGenerationMethod::Cholesky => fgn_cholesky::fgn(hurst, n - 1, t.unwrap_or(1.0)),
    };
    let mut fbm = Array1::<f64>::zeros(n);

    for i in 1..n {
        fbm[i] = fbm[i - 1] + fgn[i - 1];
    }

    fbm.to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::statistics::fractal_dim::higuchi_fd;
    use crate::utils::NoiseGenerationMethod;

    #[test]
    fn test_fbm() {
        let (hurst, n) = (0.7, 10000);

        let path = fbm(hurst, n, None, Some(NoiseGenerationMethod::Fft));
        assert_eq!(path.len(), n);
        assert_eq!(path[0], 0.0);

        let h = higuchi_fd(&path, 10);
        println!("Higuchi FD: {}", h);
        assert!((2.0 - h) < 10e-1);

        let path = fbm(hurst, n, None, Some(NoiseGenerationMethod::Cholesky));
        assert_eq!(path.len(), n);

        let h = higuchi_fd(&path, 10);
        assert!((2.0 - h) < 10e-1);
    }
}
