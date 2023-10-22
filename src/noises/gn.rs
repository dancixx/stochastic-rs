use ndarray::Array1;
use rand::{thread_rng, Rng};
use rand_distr::StandardNormal;

pub fn gn(n: usize, t: Option<f64>) -> Vec<f64> {
    let sqrt_dt = (t.unwrap_or(1.0) / n as f64).sqrt();
    let noise = thread_rng()
        .sample_iter::<f64, StandardNormal>(StandardNormal)
        .take(n)
        .collect();
    let gn = Array1::<f64>::from_vec(noise);

    (gn * sqrt_dt).to_vec()
}
