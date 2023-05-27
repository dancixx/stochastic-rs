use stochastic_rs::processes::{bm, fbm};

fn main() {
    let fbm = fbm::fbm_cholesky(1000, 0.7, Some(2));
    println!("fbm: {:?}", fbm[0]);

    let bm = bm::bm(10000, None);
    println!("bm: {:?}", bm[0]);
}
