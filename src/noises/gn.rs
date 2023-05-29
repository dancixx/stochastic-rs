use nalgebra::RowDVector;
use rand::{thread_rng, Rng};
use rand_distr::StandardNormal;

pub fn gn(n: usize, t: usize) -> RowDVector<f64> {
    let sqrt_dt = (t as f64 / n as f64).sqrt();
    let noise = thread_rng()
        .sample_iter::<f64, StandardNormal>(StandardNormal)
        .take(n)
        .collect();
    let gn = RowDVector::<f64>::from_vec(noise);

    gn * sqrt_dt
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
