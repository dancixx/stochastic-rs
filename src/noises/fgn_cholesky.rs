use nalgebra::{DMatrix, DVector, Dim, Dyn, RowDVector};
use rand_distr::{Distribution, Normal};
use std::cmp::Ordering::{Equal, Greater, Less};

fn afc_vector(n: usize, hurst: f64) -> RowDVector<f64> {
    let mut v = RowDVector::<f64>::zeros(n);
    v[0] = 1.0;

    for i in 1..n {
        let idx = i as f64;

        v[i] = 0.5
            * ((idx + 1.0).powf(2.0 * hurst) - 2.0 * idx.powf(2.0 * hurst)
                + (idx - 1.0).powf(2.0 * hurst))
    }

    v
}

fn afc_matrix_sqrt(n: usize, hurst: f64) -> DMatrix<f64> {
    let acf_v = afc_vector(n, hurst);
    let dim = Dyn::from_usize(n);
    let mut m = DMatrix::<f64>::zeros_generic(dim, dim);

    for i in 0..n {
        for j in 0..n {
            // use cmp and match instead of if-else
            //  match i.cmp(&j) {
            //     Equal => m[(i, j)] = acf_v[0],
            //     Less => m[(i, j)] = acf_v[i - j],
            //     Greater => continue,
            // }
            match i.cmp(&j) {
                Equal => m[(i, j)] = acf_v[0],
                Greater => m[(i, j)] = acf_v[i - j],
                Less => continue,
            }

            // if i == j {
            //     m[(i, j)] = acf_v[0];
            // } else if j < i {
            //     m[(i, j)] = acf_v[i - j];
            // }
        }
    }

    m.cholesky().unwrap().l()
}

pub fn fgn(n: usize, hurst: f64) -> RowDVector<f64> {
    let acf_sqrt = afc_matrix_sqrt(n, hurst);
    let normal = Normal::new(0.0, 1.0).unwrap();
    let mut rng = rand::thread_rng();
    let mut noise = DVector::<f64>::zeros(n);

    for i in 0..n {
        noise[i] = normal.sample(&mut rng);
    }

    (acf_sqrt * noise).transpose() * (n as f64).powf(-hurst)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_afc_vector() {
        let n = 1000;
        let hurst = 0.7;
        let v = afc_vector(n, hurst);
        assert_eq!(v.len(), n);
    }

    #[test]
    fn test_afc_matrix_sqrt() {
        let n = 1000;
        let hurst = 0.7;
        let m = afc_matrix_sqrt(n, hurst);
        assert_eq!(m.nrows(), n);
        assert_eq!(m.ncols(), n);
    }

    #[test]
    fn test_fgn() {
        let n = 1000;
        let hurst = 0.7;
        let fgn = fgn(n, hurst);
        assert_eq!(fgn.len(), n);
    }
}
