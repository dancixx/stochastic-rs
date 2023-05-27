use nalgebra::RowDVector;
use rand_distr::{Distribution, Normal};

pub fn gn(n: usize, t: usize) -> RowDVector<f64> {
    let sqrt_dt = (t as f64 / n as f64).sqrt();
    let mut gn = RowDVector::<f64>::zeros(n);

    let normal = Normal::new(0.0, 1.0).unwrap();
    let mut rng = rand::thread_rng();

    for i in 0..(n - 1) {
        gn[i] = sqrt_dt * normal.sample(&mut rng);
    }

    gn
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gn() {
        let n = 1000;
        let t = 1;
        let gn = gn(n, t);
        assert_eq!(gn.len(), n);
    }
}
