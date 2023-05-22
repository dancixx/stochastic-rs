use crate::noises::gn;

pub fn bm(n: usize, t: usize) -> [Vec<f64>; 2] {
    let gn = gn::gn(n, t);
    let mut bm = vec![0.0; n];

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
        let bm = bm(n, t);
        assert_eq!(bm.len(), n);
    }
}
