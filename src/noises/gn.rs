use rand_distr::{Distribution, Normal};

pub fn gn(n: usize, t: usize) -> Vec<f64> {
    let dt = (t / n) as f64;
    let sqrt_dt = dt.sqrt();
    let mut gn = vec![0.0; n - 1];

    let normal = Normal::new(0.0, 1.0).unwrap();
    let mut rng = rand::thread_rng();

    for i in 1..(n - 1) {
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
        assert_eq!(gn.len(), n - 1);
    }
}
