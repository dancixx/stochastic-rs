use crate::noises::fgn_cholesky::fgn;
use nalgebra::RowDVector;

pub fn fbm_cholesky(n: usize, hurst: f64, t: Option<usize>) -> [RowDVector<f64>; 2] {
    let fgn = fgn(n - 1, hurst);
    let mut fbm = RowDVector::<f64>::zeros(n);
    fbm[0] = 0.0;

    for i in 1..n {
        fbm[i] = fbm[i - 1] + fgn[i - 1];
    }

    let fbm = fbm * (t.unwrap_or(1) as f64).powf(hurst);

    [fbm, fgn]
}

#[allow(dead_code)]
fn fbm_fft() {
    todo!()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fbm_cholesky() {
        let n = 1000;
        let t = 1;
        let hurst = 0.7;
        let [path, incs] = fbm_cholesky(n, hurst, Some(t));
        println!("{:?}", path.len());
        println!("{:?}", incs.len());
        assert_eq!(path.len(), n);
        assert_eq!(incs.len(), n - 1);
    }
}
