use nalgebra::RowDVector;

use crate::noises::gn;

pub fn bm(n: usize, t: Option<usize>) -> [RowDVector<f64>; 2] {
    let gn = gn::gn(n - 1, t.unwrap_or(1));
    let mut bm = RowDVector::<f64>::zeros(n);
    bm[0] = 0.0;

    for i in 1..n {
        bm[i] = bm[i - 1] + gn[i - 1];
    }

    [bm, gn]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bm() {
        let n = 1000;
        let t = 1;
        let [path, incs] = bm(n, Some(t));
        assert_eq!(path.len(), n);
        assert_eq!(incs.len(), n - 1);
    }
}
