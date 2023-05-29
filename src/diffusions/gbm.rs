use crate::noises::gn::gn;
use nalgebra::DVector;

pub fn gbm(mu: f64, sigma: f64, n: usize, t: Option<usize>, x0: Option<f64>) -> DVector<f64> {
    let noise = gn(n - 1, t.unwrap_or(1));
    let dt = t.unwrap_or(1) as f64 / n as f64;
    let mut s = DVector::<f64>::zeros(n);
    s[0] = x0.unwrap_or(100.0);

    for i in 1..n {
        let ds = (mu * dt + sigma * noise[i - 1]) * s[i - 1];
        s[i] = s[i - 1] + ds;
    }

    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gbm() {
        let mu = 0.1;
        let sigma = 0.2;
        let n = 1000;
        let t = 1;
        let x0 = 100.0;
        let gbm = gbm(mu, sigma, n, Some(t), Some(x0));
        assert_eq!(gbm.len(), n);
    }
}
