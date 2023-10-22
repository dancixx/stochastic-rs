use ndarray::Array1;

use crate::noises::gn;

pub fn bm(n: usize, t: Option<f64>) -> Vec<f64> {
    let gn = gn::gn(n - 1, Some(t.unwrap_or(1.0)));
    let mut bm = Array1::<f64>::zeros(n);

    for i in 1..n {
        bm[i] = bm[i - 1] + gn[i - 1];
    }

    bm.to_vec()
}
